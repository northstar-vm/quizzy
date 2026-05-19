import os
import requests
from dotenv import load_dotenv
import numpy as np
from huggingface_hub import InferenceClient

load_dotenv()

HF_API_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

client = InferenceClient(model=EMBEDDING_MODEL, token=HF_API_TOKEN)

cached_clusters = None
cached_cluster_names = None
clusters_dirty = True
quiz_count_when_clustered = 0

clusters = None
cluster_names = None


def generate_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    cleaned = [t.strip() for t in texts if t and t.strip()]
    embeddings = client.feature_extraction(cleaned)
    
    normalized = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        normalized.append((np.array(emb) / norm).tolist())
    
    return normalized


def build_quiz_clustering_text(quiz_item):
    if isinstance(quiz_item, str):
        return quiz_item.strip()

    if not isinstance(quiz_item, dict):
        return ""

    parts = []

    title = str(quiz_item.get("title", "")).strip()
    if title:
        parts.append(f"Title: {title}")

    for question in quiz_item.get("questions", []) or []:
        question_text = str(question.get("question_text", "")).strip()
        if question_text:
            parts.append(f"Question: {question_text}")

        correct_answers = []
        for answer in question.get("answers", []) or []:
            if answer.get("is_correct"):
                answer_text = str(answer.get("answer_text", "")).strip()
                if answer_text:
                    correct_answers.append(answer_text)

        for answer_text in correct_answers:
            parts.append(f"Correct answer: {answer_text}")

    return "\n".join(parts)


def build_quiz_clustering_texts(quiz_items):
    return [build_quiz_clustering_text(item) for item in quiz_items]


def run_full_clustering():
    global clusters, cluster_names, quiz_count_when_clustered

    try:
        print(f"Starting clustering...")
        from core.database import get_db
        db = get_db()

        quizzes = list(db.quizzes.find({"userId": {"$ne": None}}))
        titles = [q.get('title', 'Untitled Quiz') for q in quizzes]
        clustering_texts = build_quiz_clustering_texts(quizzes)

        if not titles:
            return 0
            
        if quiz_count_when_clustered == len(titles) and clusters is not None:
            return len(set(clusters))

        clusters = cluster_quiz_titles(clustering_texts)
        cluster_count = len(set(clusters))

        from core.llm import get_llm_client
        llm_client = get_llm_client()
        cluster_names = {}

        if llm_client:
            cluster_titles = {}
            for idx, cluster_id in enumerate(clusters):
                cluster_titles.setdefault(cluster_id, []).append(titles[idx])

            for cluster_id, titles_in_cluster in cluster_titles.items():
                try:
                    prompt = f"Give a VERY SHORT category name for these quiz titles. ONLY RETURN 1 TO 3 WORDS MAXIMUM. ABSOLUTELY NO EXTRA TEXT, NO DASHES, NO PUNCTUATION, JUST THE NAME:\n"
                    prompt += "\n".join([f"- {t}" for t in titles_in_cluster])
                    response = llm_client.invoke(prompt)
                    if response.content:
                        name = response.content.strip().strip('"\'').title()
                        name_words = name.split()
                        if len(name_words) > 3:
                            name = ' '.join(name_words[:3])
                        cluster_names[cluster_id] = name
                except Exception as e:
                    pass
        
        quiz_count_when_clustered = len(titles)
        print(f"Clustering completed, {cluster_count} clusters created")
        
        return cluster_count

    except Exception as e:
        print(f"Clustering FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


def cluster_quiz_titles(quiz_items):
    if len(quiz_items) <= 1:
        return [0] * len(quiz_items)

    clustering_texts = build_quiz_clustering_texts(quiz_items)
    
    embeddings = generate_embeddings(clustering_texts)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    X = np.array(embeddings)
    
    max_k = min(10, len(clustering_texts) - 1)
    candidates = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        cluster_sizes = np.bincount(labels, minlength=k)
        smallest_cluster = cluster_sizes.min()
        largest_cluster_share = cluster_sizes.max() / len(clustering_texts)

        if smallest_cluster < 2 and len(clustering_texts) >= 6:
            continue

        score = silhouette_score(X, labels)
        if largest_cluster_share > 0.7:
            score -= 0.15
        score -= k * 0.01
        candidates.append((score, k))

    optimal_k = max(candidates)[1] if candidates else 1
    
    if optimal_k == 1:
        return [0] * len(clustering_texts)

    final_kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    clusters = final_kmeans.fit_predict(X)
    
    return clusters.tolist()
