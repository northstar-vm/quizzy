
import React, { useState, useEffect } from "react";
import { Button, Offcanvas, ListGroup, Dropdown, Form, Spinner, Alert } from "react-bootstrap";
import { useNavigate, useLocation } from "react-router-dom";
import { GoogleLogin } from '@react-oauth/google';
import { CredentialResponse } from '@react-oauth/google';
import { QuizData, User } from "../interfaces/interfaces";


interface ContextMenuState {
    visible: boolean;
    x: number;
    y: number;
    quizId: string | null;
    quizTitle: string | null;
    isOwned: boolean;
}


interface Props {
  guestQuizList: QuizData[];
  publicQuizList: QuizData[];
  userQuizList: QuizData[];
  selectedQuizId: string | null;
  onSelectTitleItem: (id: string) => void;
  onDeleteQuiz: (id: string, title: string) => void;

  shuffleQuestions: boolean;
  shuffleAnswers: boolean;
  onShuffleQuestionsToggle: () => void;
  onShuffleAnswersToggle: () => void;

  currentUser: User | null;
  authLoading: boolean;
  onLoginSuccess: (credentialResponse: CredentialResponse) => Promise<void>;
  onLoginError: () => void;
  onLogout: () => Promise<void>;
  loginApiError: string | null;
}

const QuizManager = ({
    guestQuizList,
    publicQuizList,
    userQuizList,
    selectedQuizId,
    onSelectTitleItem,
    onDeleteQuiz,
    shuffleQuestions,
    shuffleAnswers,
    onShuffleQuestionsToggle,
    onShuffleAnswersToggle,
    currentUser,
    authLoading,
    onLoginSuccess,
    onLoginError,
    onLogout,
    loginApiError,
}: Props) => {
  const [showOffcanvas, setShowOffcanvas] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const [contextMenu, setContextMenu] = useState<ContextMenuState>({ visible: false, x: 0, y: 0, quizId: null, quizTitle: null, isOwned: false });
  const [isLoggingOut, setIsLoggingOut] = useState(false);


  const [clusterizeEnabled, setClusterizeEnabled] = useState(false);
  const [clusters, setClusters] = useState<number[] | null>(null);
  const [clusterNames, setClusterNames] = useState<{[key: number]: string} | null>(null);
  const canClusterize = !!currentUser && userQuizList.length >= 2;
  const clusterizeLabel = !currentUser
    ? "Login to cluster quizzes"
    : userQuizList.length < 2
      ? "Need at least 2 quizzes"
      : "Clusterize Quizzes";

  const buildClusterPayload = (quiz: QuizData) => ({
    title: quiz.title,
    questions: quiz.questions.map((question) => ({
      question_text: question.question_text,
      answers: question.answers
        .filter((answer) => answer.is_correct)
        .map((answer) => ({
          answer_text: answer.answer_text,
          is_correct: true,
        })),
    })),
  });


  useEffect(() => {

    if (!clusterizeEnabled) return;


    const activeList = currentUser ? userQuizList : guestQuizList;
    if (activeList.length === 0) {
      setClusters(null);
      setClusterNames(null);
      return;
    }


    const timer = setTimeout(async () => {

      try {

        let response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/cluster-quizzes/extract`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ quizzes: activeList.map(buildClusterPayload) })
        });
        let data = response.ok ? await response.json() : null;

        if (!response.ok || data?.status === "missing") {
          response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/cluster-quizzes/clusterize`, {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ quizzes: activeList.map(buildClusterPayload) })
          });
          data = response.ok ? await response.json() : null;
        }

        if (response.ok && data?.clusters && data.clusters.length > 0 && Object.keys(data.names).length > 0) {
          setClusters(data.clusters);
          setClusterNames(data.names);
        }
      } catch (err) {
        console.error("Clustering failed:", err);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [currentUser, userQuizList, guestQuizList, clusterizeEnabled]);


  const handleCloseOffcanvas = () => setShowOffcanvas(false);
  const handleShowOffcanvas = () => setShowOffcanvas(true);
  const closeContextMenu = () => setContextMenu({ ...contextMenu, visible: false });


  const handleContextMenu = (event: React.MouseEvent<HTMLElement>, quiz: QuizData, isOwned: boolean) => {
      event.preventDefault();

      if (currentUser && isOwned) {
           setContextMenu({ visible: true, x: event.pageX, y: event.pageY, quizId: quiz.id, quizTitle: quiz.title, isOwned });
      }

  };


  useEffect(() => {
      if (!contextMenu.visible) return;
      const handleClickOutside = (event: MouseEvent) => {
         const targetElement = event.target as Element;

         if (targetElement && !targetElement.closest('.dropdown-menu')) {
            closeContextMenu();
        }
      };
      document.addEventListener("click", handleClickOutside, true);

      return () => document.removeEventListener("click", handleClickOutside, true);
  }, [contextMenu.visible]);


  const handleCreateClick = () => {
      navigate('/create');


  };


  const handleQuizSelect = (id: string) => {
      onSelectTitleItem(id);

      if (location.pathname !== '/') {
          navigate('/');
      }


  };


  const handleEditClick = () => {
      if (contextMenu.quizId && contextMenu.isOwned && currentUser) {
          navigate(`/edit/${contextMenu.quizId}`);
          closeContextMenu();
          handleCloseOffcanvas();
      } else {
          console.warn("Edit attempted without ownership or login.");
          closeContextMenu();
      }
  };


  const handleDeleteClick = () => {
      if (contextMenu.quizId && contextMenu.quizTitle && contextMenu.isOwned && currentUser) {
          onDeleteQuiz(contextMenu.quizId, contextMenu.quizTitle);
          closeContextMenu();

      } else {
          console.warn("Delete attempted without ownership or login.");
          closeContextMenu();
      }
  };


  const handleLogoutClick = async () => {
      setIsLoggingOut(true);
      try {
          await onLogout();

      } catch (err) {

          console.error("Logout failed in manager:", err);
      } finally {
          setIsLoggingOut(false);
          handleCloseOffcanvas();
      }
  };


  const renderQuizList = (list: QuizData[], title: string, type: 'guest' | 'user' | 'public', clusterOffset: number = 0) => {
    if (list.length === 0) {
      return (
        <ListGroup variant="flush">
           {list.length === 0 && type === 'guest' && !currentUser && <ListGroup.Item disabled className="text-muted small fst-italic">(No temporary quizzes)</ListGroup.Item>}
           {list.length === 0 && type === 'user' && currentUser && <ListGroup.Item disabled className="text-muted small fst-italic">(No quizzes created yet)</ListGroup.Item>}
           {list.length === 0 && type === 'public' && <ListGroup.Item disabled className="text-muted small fst-italic">(No public quizzes available)</ListGroup.Item>}
        </ListGroup>
      );
    }


    if (clusterizeEnabled && clusters && clusters.length > 0 && clusterNames && Object.keys(clusterNames).length > 0 && clusterOffset >= 0) {

      const clusterGroups: { [key: number]: QuizData[] } = {};

      list.forEach((quiz, idx) => {
        const clusterIdx = clusters[clusterOffset + idx];
        if (!clusterGroups[clusterIdx]) {
          clusterGroups[clusterIdx] = [];
        }
        clusterGroups[clusterIdx].push(quiz);
      });

      return (
        <>
          <h5 className="mt-3 mb-1 text-muted">{title} <span className="badge bg-light text-muted ms-1">{list.length}</span></h5>
          {Object.entries(clusterGroups)
            .sort(([a], [b]) => parseInt(a) - parseInt(b))
            .map(([clusterId, quizzes]) => (
              <div key={`cluster-${type}-${clusterId}`}>
                <h6 className="mt-2 mb-1 text-primary small fw-bold">
                  {clusterNames ? clusterNames[parseInt(clusterId)] : `Cluster ${parseInt(clusterId) + 1}`}
                  <span className="text-muted fw-normal ms-1">({quizzes.length} quizzes)</span>
                </h6>
                <ListGroup variant="flush" className="mb-2">
                  {quizzes.map((quiz) => {
                    const canEditOrDelete = type === 'user' && !!currentUser;
                    return (
                      <ListGroup.Item
                        action
                        active={selectedQuizId === quiz.id}
                        key={`${type}-${quiz.id}`}
                        onClick={() => handleQuizSelect(quiz.id)}
                        onContextMenu={canEditOrDelete ? (e) => handleContextMenu(e, quiz, true) : undefined}
                        style={{ cursor: 'pointer', paddingLeft: '20px', paddingRight: '10px' }}
                        title={canEditOrDelete ? `${quiz.title} (Right-click for options)` : quiz.title}
                      >
                        {quiz.title}
                        {type === 'guest' && <span className="badge bg-secondary ms-2 float-end">Temporary</span>}
                      </ListGroup.Item>
                    );
                  })}
                </ListGroup>
              </div>
            ))}
        </>
      );
    }


    return (
     <>
        {list.length > 0 && <h5 className="mt-3 mb-1 text-muted">{title}</h5>}
        <ListGroup variant="flush">
          {list.map((quiz) => {
            const canEditOrDelete = type === 'user' && !!currentUser;
            return (
                <ListGroup.Item
                  action
                  active={selectedQuizId === quiz.id}
                  key={`${type}-${quiz.id}`}
                  onClick={() => handleQuizSelect(quiz.id)}
                  onContextMenu={canEditOrDelete ? (e) => handleContextMenu(e, quiz, true) : undefined}
                  style={{ cursor: 'pointer', paddingLeft: '10px', paddingRight: '10px' }}
                  title={canEditOrDelete ? `${quiz.title} (Right-click for options)` : quiz.title}
                >
                  {quiz.title}
                  {type === 'guest' && <span className="badge bg-secondary ms-2 float-end">Temporary</span>}
                </ListGroup.Item>
            );
          })}
        </ListGroup>
     </>
    );
  };


  return (
    <>

      <Button variant="primary" style={{ position: "fixed", top: "1rem", left: "1rem", zIndex: 0 }} onClick={handleShowOffcanvas}>
        ☰ Menu
      </Button>


      {contextMenu.visible && contextMenu.isOwned && currentUser && (
          <Dropdown.Menu show style={{ position: 'absolute', left: `${contextMenu.x}px`, top: `${contextMenu.y}px`, zIndex: 1100 }}>
              <Dropdown.Header>{contextMenu.quizTitle || "Actions"}</Dropdown.Header>
              <Dropdown.Item onClick={handleEditClick}>Edit Quiz</Dropdown.Item>
              <Dropdown.Item onClick={handleDeleteClick} className="text-danger">Delete Quiz</Dropdown.Item>
              <Dropdown.Divider />
              <Dropdown.Item onClick={closeContextMenu}>Cancel</Dropdown.Item>
          </Dropdown.Menu>
      )}


      <Offcanvas id="offcanvasQuizManager" show={showOffcanvas} onHide={handleCloseOffcanvas} placement="start" backdrop={false} scroll={true}>
        <Offcanvas.Header closeButton>
          <Offcanvas.Title>Quizzy Menu</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body className="d-flex flex-column">


          <div className="mb-3 border-bottom pb-3">

              {authLoading ? (
                   <div className="text-center"><Spinner animation="border" size="sm" /> Loading User...</div>
              ) : currentUser ? (

                  <div className="d-flex flex-column align-items-center">
                      {currentUser.picture && (
                          <img src={currentUser.picture} alt="User profile" referrerPolicy="no-referrer" style={{ width: '40px', height: '40px', borderRadius: '50%', marginBottom: '5px'}}/>
                      )}
                      <span className="mb-1 text-center small">Welcome, {currentUser.name}!</span>
                      <Button variant="outline-secondary" size="sm" onClick={handleLogoutClick} disabled={isLoggingOut}>
                          {isLoggingOut ? <Spinner animation="border" size="sm" /> : "Logout"}
                      </Button>


                  </div>
              ) : (

                  <div className="d-grid">
                      <GoogleLogin
                          onSuccess={onLoginSuccess}
                          onError={onLoginError}
                          useOneTap={false}
                          theme="outline"
                          size="medium"
                       />
                       {loginApiError && <Alert variant="danger" className="mt-2 p-1 text-center small">{loginApiError}</Alert>}
                  </div>
              )}
          </div>


          <Button variant="success" className="w-100 mb-3" onClick={handleCreateClick}>
            + Create New Quiz (AI)
          </Button>


          <div className="mb-3 border p-2 rounded bg-light">
            <Form.Label className="fw-bold small">Quiz Display Options</Form.Label>
            <Form.Check
                type="switch" id="shuffle-questions-check" label="Shuffle Questions"
                checked={shuffleQuestions} onChange={onShuffleQuestionsToggle} className="small"
            />
            <Form.Check
                type="switch" id="shuffle-answers-check" label="Shuffle Answers"
                checked={shuffleAnswers} onChange={onShuffleAnswersToggle} className="small"
            />
            <hr className="my-2"/>
            <Form.Check
                type="switch"
                id="clusterize-check"
                label={
                  (clusterizeEnabled && !(clusters && clusterNames))
                    ? <>
                        <Spinner animation="border" size="sm" className="me-2" />
                        Initializing clusters...
                      </>
                    : clusterizeLabel
                }
                checked={clusterizeEnabled}
                disabled={!canClusterize || (clusterizeEnabled && !(clusters && clusterNames))}
                onChange={() => {
                    if (!canClusterize) return;
                    const newState = !clusterizeEnabled;
                    setClusterizeEnabled(newState);

                    if (!newState) {
                       setClusters(null);
                       setClusterNames(null);
                    }
                }}
                className="small"
            />
          </div>


           <div style={{ flexGrow: 1, overflowY: 'auto', borderTop: '1px solid #eee', paddingTop: '10px' }}>

               {!currentUser && renderQuizList(guestQuizList, "Temporary Quizzes", 'guest', 0)}
               {currentUser && renderQuizList(userQuizList, "My Quizzes", 'user', 0)}

               {renderQuizList(publicQuizList, "Public Quizzes", 'public', -1)}


               {!authLoading && guestQuizList.length === 0 && userQuizList.length === 0 && publicQuizList.length === 0 && (
                   <p className="text-center text-muted mt-3">No quizzes found. Create one!</p>
               )}
            </div>

        </Offcanvas.Body>
      </Offcanvas>
    </>
  );
};

export default QuizManager;
