"""
Tests for external API connections: Groq LLM, HuggingFace Embeddings
"""
import io
import json
import sys
import os
from types import SimpleNamespace
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import pytest
from flask import Flask
import api.quizzes as quiz_api
import api.recommendations as recommendation_api
import api.chat as chat_api
import core.quiz_generation as quiz_generation
from api.quizzes import quiz_routes
from api.recommendations import recommendation_routes
from api.chat import chat_routes
from core.pdf_processor import PDFProcessor
from core.quiz_generation import (
    create_distractor_generation_prompt,
    create_question_plan_prompt,
    generate_questions_from_plans,
    normalize_generated_questions,
    normalize_question_plans,
    parse_llm_json,
)
from core.llm import (
    get_available_groq_models,
    get_llm_client,
)
from core.embeddings import (
    EMBEDDING_DIMENSION,
    build_quiz_clustering_text,
    generate_embeddings,
)


RUN_EXTERNAL_API_TESTS = os.environ.get("RUN_EXTERNAL_API_TESTS") == "1"


requires_external_apis = pytest.mark.skipif(
    not RUN_EXTERNAL_API_TESTS,
    reason="External API tests are disabled. Set RUN_EXTERNAL_API_TESTS=1 to run them.",
)


def _sample_question():
    return {
        "id": "question-1",
        "type": "multiple_choice",
        "question_text": "What is 2 + 2?",
        "answers": [
            {"id": "answer-1", "answer_text": "4", "is_correct": True},
            {"id": "answer-2", "answer_text": "3", "is_correct": False},
            {"id": "answer-3", "answer_text": "5", "is_correct": False},
            {"id": "answer-4", "answer_text": "6", "is_correct": False},
        ],
    }


def _stream_events(response):
    body = response.get_data(as_text=True)
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


class _FailingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise RuntimeError("model_decommissioned")


class _FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def find(self, query):
        def matches(doc):
            return all(doc.get(key) == value for key, value in query.items())

        return [doc for doc in self.documents if matches(doc)]


class _FakeDb:
    def __init__(self, quiz_documents):
        self.quizzes = _FakeCollection(quiz_documents)


def _create_app(*blueprints):
    app = Flask(__name__)
    for blueprint in blueprints:
        app.register_blueprint(blueprint)
    return app


class TestExternalAPIs:
    
    def setup_class(self):
        """Load environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
    
    def test_available_groq_models_are_curated(self):
        models = get_available_groq_models()

        assert len(models) >= 1
        assert any(model["id"] == "llama-3.3-70b-versatile" for model in models)
        assert all("context_window" in model for model in models)

    def test_question_plan_prompt_prefers_adapting_existing_tasks(self):
        prompt = create_question_plan_prompt(
            "1. Solve 2 + 2. 2. Name the capital of Lithuania.",
            source_type="pdf",
            requested_question_count=6,
            difficulty=4,
            language="Lithuanian",
        )

        assert "EXACTLY 6 question plans" in prompt
        assert "adapt them into quiz question plans" in prompt
        assert "existing_task" in prompt
        assert "Lithuanian" in prompt

    def test_distractor_prompt_uses_whole_batch(self):
        prompt = create_distractor_generation_prompt(
            [
                {
                    "question_text": "What is 2 + 2?",
                    "correct_answer": "4",
                    "source_reference": "Page 1",
                    "concept": "Addition",
                    "origin": "existing_task",
                }
            ],
            difficulty=5,
            language="English",
        )

        assert "QUESTION PLANS" in prompt
        assert "exactly 3 incorrect distractors" in prompt
        assert "expert-level" in prompt
        assert '"questions"' in prompt
        assert "valid JSON accepted by json.loads" in prompt
        assert "Do not wrap the response in Markdown" in prompt

    def test_parse_llm_json_handles_fenced_json(self):
        parsed = parse_llm_json('```json\n{"question_plans": []}\n```')

        assert parsed == {"question_plans": []}

    def test_parse_llm_json_handles_json_like_model_output(self):
        parsed = parse_llm_json("""
        Here is the quiz:
        ```json
        {
          'questions': [
            {
              'question_text': 'What is 2 + 2?',
              'answers': [
                {'answer_text': '4', 'is_correct': True},
                {'answer_text': '3', 'is_correct': False},
              ],
            }
          ],
        }
        ```
        """)

        assert parsed["questions"][0]["answers"][0]["is_correct"] is True

    def test_parse_llm_json_uses_first_valid_json_object(self):
        parsed = parse_llm_json("""
        {"questions": []}
        Note: I used the requested format.
        {"ignored": true}
        """)

        assert parsed == {"questions": []}

    def test_parse_llm_json_rejects_malformed_json(self):
        with pytest.raises(ValueError):
            parse_llm_json("not json at all")

    def test_normalize_question_plans_keeps_required_contract(self):
        plans = normalize_question_plans({
            "question_plans": [{
                "question": "What is 2 + 2?",
                "answer": "4",
                "source_reference": "Page 1",
                "concept": "Addition",
                "origin": "unknown",
            }]
        })

        assert plans == [{
            "question_text": "What is 2 + 2?",
            "correct_answer": "4",
            "source_reference": "Page 1",
            "concept": "Addition",
            "origin": "document_content",
        }]

    def test_normalize_generated_questions_adds_ids_and_validates_answers(self):
        questions = normalize_generated_questions({
            "questions": [{
                "question_text": "What is 2 + 2?",
                "answers": [
                    {"answer_text": "4", "is_correct": True},
                    {"answer_text": "3", "is_correct": False},
                    {"answer_text": "5", "is_correct": False},
                    {"answer_text": "6", "is_correct": False},
                ]
            }]
        })

        assert len(questions) == 1
        assert questions[0]["id"]
        assert questions[0]["type"] == "multiple_choice"
        assert len(questions[0]["answers"]) == 4
        assert sum(answer["is_correct"] for answer in questions[0]["answers"]) == 1
        assert all(answer["id"] for answer in questions[0]["answers"])

    def test_normalize_generated_questions_accepts_options_and_correct_answer(self):
        questions = normalize_generated_questions({
            "questions": [{
                "prompt": "What is 2 + 2?",
                "options": ["3", "4", "5", "6"],
                "correctAnswer": "4",
            }]
        })

        assert len(questions) == 1
        assert questions[0]["question_text"] == "What is 2 + 2?"
        assert sum(answer["is_correct"] for answer in questions[0]["answers"]) == 1
        assert questions[0]["answers"][0]["answer_text"] == "4"

    def test_normalize_generated_questions_accepts_choice_labels(self):
        questions = normalize_generated_questions({
            "quiz": {
                "questions": [{
                    "stem": "What is the capital of Lithuania?",
                    "choices": {
                        "A": "Riga",
                        "B": "Vilnius",
                        "C": "Tallinn",
                        "D": "Warsaw",
                    },
                    "correct_option": "B",
                }]
            }
        })

        assert len(questions) == 1
        assert questions[0]["answers"][0]["answer_text"] == "Vilnius"
        assert sum(answer["is_correct"] for answer in questions[0]["answers"]) == 1

    def test_normalize_generated_questions_treats_false_string_as_false(self):
        questions = normalize_generated_questions({
            "questions": [{
                "question_text": "What is 2 + 2?",
                "answers": [
                    {"answer_text": "4", "is_correct": "true"},
                    {"answer_text": "3", "is_correct": "false"},
                    {"answer_text": "5", "is_correct": "false"},
                    {"answer_text": "6", "is_correct": "false"},
                ]
            }]
        })

        assert sum(answer["is_correct"] for answer in questions[0]["answers"]) == 1
        assert questions[0]["answers"][0]["answer_text"] == "4"

    def test_generate_questions_from_plans_retries_invalid_json(self, monkeypatch):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content="I cannot provide JSON for that.")
                return SimpleNamespace(content=json.dumps({
                    "questions": [{
                        "question_text": "What is 2 + 2?",
                        "answers": [
                            {"answer_text": "4", "is_correct": True},
                            {"answer_text": "3", "is_correct": False},
                            {"answer_text": "5", "is_correct": False},
                            {"answer_text": "6", "is_correct": False},
                        ],
                    }]
                }))

        fake_llm = FakeLLM()
        monkeypatch.setattr("core.quiz_generation.get_llm_client", lambda **_: fake_llm)

        questions = generate_questions_from_plans(
            [{
                "question_text": "What is 2 + 2?",
                "correct_answer": "4",
                "source_reference": "Page 1",
                "concept": "Addition",
                "origin": "existing_task",
            }],
            difficulty=3,
            language="English",
        )

        assert fake_llm.calls == 2
        assert questions[0]["question_text"] == "What is 2 + 2?"

    def test_generate_questions_from_plans_retries_unusable_json(self, monkeypatch):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"questions": [{"question_text": "No answers"}]}))
                return SimpleNamespace(content=json.dumps({
                    "questions": [{
                        "question_text": "What is 2 + 2?",
                        "options": ["3", "4", "5", "6"],
                        "correctAnswer": "4",
                    }]
                }))

        fake_llm = FakeLLM()
        monkeypatch.setattr("core.quiz_generation.get_llm_client", lambda **_: fake_llm)

        questions = generate_questions_from_plans(
            [{
                "question_text": "What is 2 + 2?",
                "correct_answer": "4",
                "source_reference": "Page 1",
                "concept": "Addition",
                "origin": "existing_task",
            }],
            difficulty=3,
            language="English",
        )

        assert fake_llm.calls == 2
        assert questions[0]["answers"][0]["answer_text"] == "4"

    def test_generate_questions_from_plans_batches_large_requests(self, monkeypatch):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                batch_size = 5 if self.calls in {1, 2} else 2
                return SimpleNamespace(content=json.dumps({
                    "questions": [
                        {
                            "question_text": f"Question {self.calls}-{index}",
                            "answers": [
                                {"answer_text": "Correct", "is_correct": True},
                                {"answer_text": "Wrong 1", "is_correct": False},
                                {"answer_text": "Wrong 2", "is_correct": False},
                                {"answer_text": "Wrong 3", "is_correct": False},
                            ],
                        }
                        for index in range(batch_size)
                    ]
                }))

        fake_llm = FakeLLM()
        monkeypatch.setattr("core.quiz_generation.get_llm_client", lambda **_: fake_llm)

        plans = [
            {
                "question_text": f"Question plan {index}",
                "correct_answer": "Correct",
                "source_reference": "Topic",
                "concept": "Batching",
                "origin": "topic_generation",
            }
            for index in range(12)
        ]

        questions = generate_questions_from_plans(plans, difficulty=3, language="English")

        assert fake_llm.calls == 3
        assert len(questions) == 12

    def test_generate_questions_from_plans_retries_repeated_rate_limits(self, monkeypatch):
        class RateLimitError(Exception):
            pass

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                if self.calls < 3:
                    raise RateLimitError("Rate limit reached. Please try again in 0.01s.")
                return SimpleNamespace(content=json.dumps({
                    "questions": [{
                        "question_text": "What is 2 + 2?",
                        "answers": [
                            {"answer_text": "4", "is_correct": True},
                            {"answer_text": "3", "is_correct": False},
                            {"answer_text": "5", "is_correct": False},
                            {"answer_text": "6", "is_correct": False},
                        ],
                    }]
                }))

        fake_llm = FakeLLM()
        sleeps = []
        monkeypatch.setattr("core.quiz_generation.get_llm_client", lambda **_: fake_llm)
        monkeypatch.setattr(quiz_generation.time, "sleep", sleeps.append)

        questions = generate_questions_from_plans(
            [{
                "question_text": "What is 2 + 2?",
                "correct_answer": "4",
                "source_reference": "Topic",
                "concept": "Addition",
                "origin": "topic_generation",
            }],
            difficulty=3,
            language="English",
        )

        assert fake_llm.calls == 3
        assert len(sleeps) == 2
        assert questions[0]["question_text"] == "What is 2 + 2?"

    def test_generate_stream_accepts_topic_form_data(self, monkeypatch):
        captured_request = {}

        def fake_generate(generation_request, progress_queue=None):
            captured_request.update(generation_request)
            return [_sample_question()]

        monkeypatch.setattr(quiz_api, "_generate_questions_for_request", fake_generate)
        monkeypatch.setattr(quiz_api, "_save_quiz_for_user", lambda *args: None)
        monkeypatch.setattr(quiz_api, "get_current_user_db_id", lambda: None)

        app = Flask(__name__)
        app.register_blueprint(quiz_routes)

        response = app.test_client().post(
            "/api/quizzes/generate-stream",
            data={
                "title": "Math quiz",
                "topic": "addition",
                "num_questions": "2",
                "difficulty": "5",
                "language": "English",
            },
        )
        events = _stream_events(response)
        complete_event = next(event for event in events if event.get("complete"))

        assert response.status_code == 200
        assert captured_request["source_type"] == "topic"
        assert captured_request["num_questions"] == 2
        assert captured_request["difficulty"] == 5
        assert complete_event["quiz"]["source_type"] == "topic"
        assert complete_event["quiz"]["questions"] == [_sample_question()]

    def test_generate_stream_accepts_pdf_form_data(self, monkeypatch):
        captured_request = {}

        def fake_generate(generation_request, progress_queue=None):
            captured_request.update(generation_request)
            return [_sample_question()]

        monkeypatch.setattr(quiz_api, "_generate_questions_for_request", fake_generate)
        monkeypatch.setattr(quiz_api, "_save_quiz_for_user", lambda *args: None)
        monkeypatch.setattr(quiz_api, "get_current_user_db_id", lambda: None)

        app = Flask(__name__)
        app.register_blueprint(quiz_routes)

        response = app.test_client().post(
            "/api/quizzes/generate-stream",
            data={
                "title": "PDF quiz",
                "topic": "adapt existing tasks",
                "num_questions": "3",
                "difficulty": "4",
                "language": "Lithuanian",
                "pdf": (io.BytesIO(b"%PDF-1.4 fake"), "tasks.pdf"),
            },
        )
        events = _stream_events(response)
        complete_event = next(event for event in events if event.get("complete"))

        assert response.status_code == 200
        assert captured_request["source_type"] == "pdf"
        assert captured_request["source_document"] == "tasks.pdf"
        assert captured_request["pdf_bytes"] == b"%PDF-1.4 fake"
        assert captured_request["difficulty"] == 4
        assert complete_event["quiz"]["source_type"] == "pdf"
        assert complete_event["quiz"]["source_document"] == "tasks.pdf"

    def test_generate_stream_requires_explicit_question_count(self, monkeypatch):
        monkeypatch.setattr(quiz_api, "get_current_user_db_id", lambda: None)

        app = _create_app(quiz_routes)

        response = app.test_client().post(
            "/api/quizzes/generate-stream",
            data={
                "title": "Math quiz",
                "topic": "addition",
                "difficulty": "3",
                "language": "English",
            },
        )

        assert response.status_code == 400
        assert response.get_json()["error"] == "Invalid 'num_questions'."

    def test_legacy_pdf_endpoint_is_removed(self):
        app = _create_app(quiz_routes)

        response = app.test_client().post(
            "/api/quizzes/generate-from-pdf",
            data={
                "title": "PDF quiz",
                "topic": "addition",
                "num_questions": "3",
                "difficulty": "3",
                "language": "English",
                "pdf": (io.BytesIO(b"%PDF-1.4 fake"), "tasks.pdf"),
            },
        )

        assert response.status_code == 405

    def test_pdf_image_description_fails_fast(self):
        completions = _FailingCompletions()
        processor = PDFProcessor.__new__(PDFProcessor)
        processor.vision_model = "broken-vision-model"
        processor.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        with pytest.raises(RuntimeError) as exc_info:
            processor._describe_image(b"fake image bytes", 1, 2)

        message = str(exc_info.value)
        assert completions.calls == 1
        assert "Visual description failed for page 2" in message
        assert "broken-vision-model" in message
        assert "model_decommissioned" in message

    def test_pdf_image_description_does_not_retry_rate_limit(self):
        class RateLimitedCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise RuntimeError("Error code: 429 - rate_limit_exceeded. Please try again in 250ms.")

        completions = RateLimitedCompletions()
        processor = PDFProcessor.__new__(PDFProcessor)
        processor.vision_model = "rate-limited-vision-model"
        processor.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        with pytest.raises(RuntimeError) as exc_info:
            processor._describe_image(b"fake image bytes", "page", 1)

        assert completions.calls == 1
        assert "rate_limit_exceeded" in str(exc_info.value)

    def test_pdf_process_skips_failed_page_visual_description(self, monkeypatch):
        class FakePage:
            def get_text(self, sort=True):
                return "Visible PDF text"

            def get_images(self, full=True):
                return []

        class FakeDoc:
            def __iter__(self):
                return iter([FakePage()])

            def __len__(self):
                return 1

            def close(self):
                pass

        processor = PDFProcessor.__new__(PDFProcessor)
        processor.vision_model = "broken-vision-model"
        processor.max_image_size = 20 * 1024 * 1024
        monkeypatch.setattr("core.pdf_processor.fitz.open", lambda **kwargs: FakeDoc())
        monkeypatch.setattr(
            processor,
            "_describe_page_visuals",
            lambda doc, page, page_num: (_ for _ in ()).throw(RuntimeError("vision failed")),
        )

        document_text = processor.process_pdf(b"%PDF fake")

        assert "Visible PDF text" in document_text
        assert "VISUAL DESCRIPTION" not in document_text

    def test_get_models_returns_available_models(self, monkeypatch):
        monkeypatch.setattr(
            "core.llm.get_available_groq_models",
            lambda: [{"id": "llama-3.3-70b-versatile", "context_window": 128000}],
        )

        app = _create_app(quiz_routes)
        response = app.test_client().get("/api/models")

        assert response.status_code == 200
        assert response.get_json()[0]["id"] == "llama-3.3-70b-versatile"

    def test_get_quizzes_public_scope_returns_public_quizzes(self, monkeypatch):
        fake_db = _FakeDb([
            {"_id": "mongo1", "id": "quiz-public", "title": "Public quiz", "userId": None},
            {"_id": "mongo2", "id": "quiz-private", "title": "Private quiz", "userId": "user-1"},
        ])
        monkeypatch.setattr(quiz_api, "get_db", lambda: fake_db)
        monkeypatch.setattr(quiz_api, "get_current_user_db_id", lambda: None)

        app = _create_app(quiz_routes)
        response = app.test_client().get("/api/quizzes?scope=public")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload == [{"id": "quiz-public", "title": "Public quiz", "userId": None}]

    def test_get_quizzes_my_scope_requires_authentication(self, monkeypatch):
        monkeypatch.setattr(quiz_api, "get_current_user_db_id", lambda: None)
        monkeypatch.setattr(quiz_api, "get_db", lambda: _FakeDb([]))

        app = _create_app(quiz_routes)
        response = app.test_client().get("/api/quizzes?scope=my")

        assert response.status_code == 401
        assert "Authentication required" in response.get_json()["error"]

    def test_get_quizzes_rejects_invalid_scope(self, monkeypatch):
        monkeypatch.setattr(quiz_api, "get_current_user_db_id", lambda: None)
        monkeypatch.setattr(quiz_api, "get_db", lambda: _FakeDb([]))

        app = _create_app(quiz_routes)
        response = app.test_client().get("/api/quizzes?scope=weird")

        assert response.status_code == 400
        assert "Invalid scope parameter" in response.get_json()["error"]

    def test_recommendations_endpoint_returns_timestamped_results(self, monkeypatch):
        monkeypatch.setattr(recommendation_api, "generate_embeddings", lambda texts: [[0.1, 0.2]])
        monkeypatch.setattr(
            recommendation_api,
            "find_similar_videos",
            lambda embedding, limit, min_score: [
                {
                    "video_id": "abc123",
                    "video_title": "Binary Search",
                    "text": "Binary search cuts the range in half.",
                    "start": 30.0,
                    "end": 40.0,
                    "score": 0.91234,
                }
            ],
        )

        app = _create_app(recommendation_routes)
        response = app.test_client().post(
            "/api/recommendations",
            json={
                "incorrect_questions": [
                    {
                        "id": "q1",
                        "question_text": "What is binary search?",
                        "correct_answer": "A divide and conquer search algorithm",
                        "user_answer": "A sorting algorithm",
                    }
                ]
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["q1"]["recommendations"][0]["youtube_url"].endswith("abc123&t=30")
        assert payload["q1"]["recommendations"][0]["relevance_score"] == 0.9123

    def test_recommendations_endpoint_merges_nearby_chunks_and_prefers_distinct_videos(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(recommendation_api, "generate_embeddings", lambda texts: [[0.1, 0.2]])

        def fake_find_similar_videos(embedding, limit, min_score):
            captured["limit"] = limit
            captured["min_score"] = min_score
            return [
                {
                    "video_id": "sleep1",
                    "video_title": "Sleep Is Your Superpower",
                    "text": "The first is regularity.",
                    "start": 858.0,
                    "end": 867.0,
                    "score": 0.84,
                },
                {
                    "video_id": "sleep1",
                    "video_title": "Sleep Is Your Superpower",
                    "text": "Go to bed at the same time.",
                    "start": 867.0,
                    "end": 875.0,
                    "score": 0.83,
                },
                {
                    "video_id": "sleep1",
                    "video_title": "Sleep Is Your Superpower",
                    "text": "Keep it cool.",
                    "start": 885.0,
                    "end": 895.0,
                    "score": 0.82,
                },
                {
                    "video_id": "focus2",
                    "video_title": "Focus Better",
                    "text": "Avoid late caffeine.",
                    "start": 120.0,
                    "end": 130.0,
                    "score": 0.81,
                },
                {
                    "video_id": "habits3",
                    "video_title": "Better Habits",
                    "text": "Keep a routine.",
                    "start": 50.0,
                    "end": 60.0,
                    "score": 0.805,
                },
            ]

        monkeypatch.setattr(recommendation_api, "find_similar_videos", fake_find_similar_videos)

        app = _create_app(recommendation_routes)
        response = app.test_client().post(
            "/api/recommendations",
            json={
                "incorrect_questions": [
                    {
                        "id": "q1",
                        "question_text": "How can I sleep better?",
                        "correct_answer": "Regular schedule",
                        "user_answer": "More naps",
                    }
                ]
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        recommendations = payload["q1"]["recommendations"]

        assert captured == {"limit": 10, "min_score": 0.8}
        assert [rec["video_id"] for rec in recommendations] == ["sleep1", "focus2", "habits3"]
        assert recommendations[0]["start_time"] == 858.0
        assert recommendations[0]["end_time"] == 895.0
        assert recommendations[0]["text"] == (
            "The first is regularity. Go to bed at the same time. Keep it cool."
        )

    def test_recommendations_endpoint_requires_question_array(self):
        app = _create_app(recommendation_routes)
        response = app.test_client().post("/api/recommendations", json={})

        assert response.status_code == 400
        assert "incorrect_questions" in response.get_json()["error"]

    def test_library_stats_endpoint_returns_stats(self, monkeypatch):
        monkeypatch.setattr(recommendation_api, "get_video_count", lambda: {"videos": 4, "chunks": 120})

        app = _create_app(recommendation_routes)
        response = app.test_client().get("/api/recommendations/library/stats")

        assert response.status_code == 200
        assert response.get_json() == {"videos": 4, "chunks": 120}

    def test_build_quiz_clustering_text_uses_questions_and_correct_answers(self):
        clustering_text = build_quiz_clustering_text({
            "title": "Python basics",
            "questions": [
                {
                    "question_text": "What keyword starts a loop?",
                    "answers": [
                        {"answer_text": "for", "is_correct": True},
                        {"answer_text": "class", "is_correct": False},
                    ],
                }
            ],
        })

        assert "Title: Python basics" in clustering_text
        assert "Question: What keyword starts a loop?" in clustering_text
        assert "Correct answer: for" in clustering_text
        assert "class" not in clustering_text

    def test_cluster_quiz_titles_avoids_single_item_clusters(self, monkeypatch):
        import core.embeddings as embeddings_api

        monkeypatch.setattr(embeddings_api, "generate_embeddings", lambda texts: [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.02, 0.98],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-0.98, -0.02],
        ])

        clusters = embeddings_api.cluster_quiz_titles([
            "Python loops",
            "Python functions",
            "SQL joins",
            "SQL selects",
            "Geography capitals",
            "Geography maps",
        ])

        cluster_sizes = {
            cluster_id: clusters.count(cluster_id)
            for cluster_id in set(clusters)
        }

        assert len(set(clusters)) == 3
        assert min(cluster_sizes.values()) >= 2

    def test_clusterize_endpoint_returns_cluster_names(self, monkeypatch):
        recommendation_api._cluster_cache.clear()
        monkeypatch.setattr(recommendation_api, "get_current_user_db_id", lambda: "user-1")
        captured_payload = {}

        def fake_cluster_quiz_titles(items):
            captured_payload["items"] = items
            return [0, 1, 1]

        monkeypatch.setattr(recommendation_api, "cluster_quiz_titles", fake_cluster_quiz_titles)

        captured_prompts = []

        class FakeLLM:
            def invoke(self, prompt):
                captured_prompts.append(prompt)
                if "SQL basics" in prompt:
                    return SimpleNamespace(content="Databases")
                return SimpleNamespace(content="Programming")

        monkeypatch.setattr(recommendation_api, "get_llm_client", lambda: FakeLLM())

        app = _create_app(recommendation_routes)
        response = app.test_client().post(
            "/api/cluster-quizzes/clusterize",
            json={
                "quizzes": [
                    {
                        "title": "Python loops",
                        "questions": [
                            {
                                "question_text": "What keyword starts a loop?",
                                "answers": [{"answer_text": "for", "is_correct": True}],
                            }
                        ],
                    },
                    {
                        "title": "SQL joins",
                        "questions": [
                            {
                                "question_text": "Which join returns matching rows?",
                                "answers": [{"answer_text": "INNER JOIN", "is_correct": True}],
                            }
                        ],
                    },
                    {
                        "title": "SQL basics",
                        "questions": [
                            {
                                "question_text": "How do you read all rows?",
                                "answers": [{"answer_text": "SELECT *", "is_correct": True}],
                            }
                        ],
                    },
                ]
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["status"] == "ready"
        assert payload["clusters"] == [0, 1, 1]
        assert payload["count"] == 2
        assert payload["names"] == {"0": "Programming", "1": "Databases"}
        assert len(captured_prompts) == 2
        assert "Question: What keyword starts a loop?" in captured_payload["items"][0]
        assert "Correct answer: for" in captured_payload["items"][0]

    def test_warm_cluster_cache_for_user_builds_cache_from_user_quizzes(self, monkeypatch):
        recommendation_api._cluster_cache.clear()
        recommendation_api._cluster_warmups.clear()

        class ImmediateThread:
            starts = 0

            def __init__(self, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self):
                ImmediateThread.starts += 1
                self.target()

        monkeypatch.setattr(recommendation_api.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(recommendation_api, "get_db", lambda: _FakeDb([
            {
                "title": "Python loops",
                "userId": "user-1",
                "questions": [{"question_text": "What starts a loop?", "answers": []}],
            },
            {
                "title": "SQL joins",
                "userId": "user-1",
                "questions": [{"question_text": "What joins rows?", "answers": []}],
            },
            {
                "title": "Other user",
                "userId": "user-2",
                "questions": [],
            },
        ]))
        monkeypatch.setattr(recommendation_api, "cluster_quiz_titles", lambda items: [0, 1])
        monkeypatch.setattr(recommendation_api, "get_llm_client", lambda: None)

        recommendation_api.warm_cluster_cache_for_user("user-1")

        cached = recommendation_api._cluster_cache["user-1"]
        assert cached["status"] == "ready"
        assert cached["clusters"] == [0, 1]
        assert cached["count"] == 2

        recommendation_api.warm_cluster_cache_for_user("user-1")
        assert ImmediateThread.starts == 1

    def test_cluster_extract_returns_missing_without_cache(self, monkeypatch):
        recommendation_api._cluster_cache.clear()
        monkeypatch.setattr(recommendation_api, "get_current_user_db_id", lambda: "user-2")

        app = _create_app(recommendation_routes)
        response = app.test_client().post(
            "/api/cluster-quizzes/extract",
            json={
                "quizzes": [
                    {"title": "Quiz A", "questions": []},
                    {"title": "Quiz B", "questions": []},
                ]
            },
        )

        assert response.status_code == 200
        assert response.get_json() == {"status": "missing"}

    def test_chat_endpoint_returns_reply(self, monkeypatch):
        captured_prompt = {}

        class FakeLLM:
            def invoke(self, prompt):
                captured_prompt["value"] = prompt
                return SimpleNamespace(content="Helpful hint")

        monkeypatch.setattr(chat_api, "get_llm_client", lambda: FakeLLM())

        app = _create_app(chat_routes)
        response = app.test_client().post(
            "/api/chat",
            json={
                "message": "Give me a hint",
                "context": {
                    "quizTitle": "Algorithms",
                    "questionText": "What is binary search?",
                    "options": ["O(n)", "O(log n)"],
                    "isReviewMode": False,
                },
            },
        )

        assert response.status_code == 200
        assert response.get_json() == {"reply": "Helpful hint"}
        assert "DO NOT REVEAL THE CORRECT ANSWER" in captured_prompt["value"]

    def test_chat_endpoint_requires_message(self, monkeypatch):
        monkeypatch.setattr(chat_api, "get_llm_client", lambda: SimpleNamespace(invoke=lambda _: None))

        app = _create_app(chat_routes)
        response = app.test_client().post("/api/chat", json={"context": {}})

        assert response.status_code == 400
        assert response.get_json()["error"] == "Missing 'message'"

    @requires_external_apis
    def test_huggingface_embedding_connection(self):
        """Test that we can connect to HuggingFace embedding API"""
        test_text = "connection test"
        embedding = generate_embeddings([test_text])
        assert embedding is not None
        assert len(embedding) == 1

    @requires_external_apis
    def test_huggingface_embedding_dimensions(self):
        """Test embeddings are generated correctly with 1024 dimensions"""
        test_text = "This is a test sentence for embedding generation"
        embedding = generate_embeddings([test_text])
        
        assert len(embedding) == 1
        assert len(embedding[0]) == EMBEDDING_DIMENSION
        assert all(isinstance(x, float) for x in embedding[0])
        
        # Verify embedding is not all zeros
        assert sum(abs(x) for x in embedding[0]) > 0.01

    @requires_external_apis
    def test_huggingface_embedding_unique(self):
        """Test different texts produce different embeddings"""
        texts = [
            "What is binary search algorithm?",
            "Explain object oriented programming",
            "How does vector database work?"
        ]
        
        embeddings = generate_embeddings(texts)
        
        assert len(embeddings) == 3
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    @requires_external_apis
    def test_groq_connection(self):
        """Test that we can connect to Groq API successfully"""
        client = get_llm_client()
        assert client is not None

    @requires_external_apis
    def test_groq_client_working(self):
        """Test Groq LLM client can respond"""
        client = get_llm_client()
        if client:
            response = client.invoke("Hello, respond with 'OK'")
            assert response is not None
            assert len(response.content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
