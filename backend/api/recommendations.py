import json
import logging
import threading
from flask import Blueprint, jsonify, request
from flask_login import login_required

from config import get_current_user_db_id, get_db
from core.embeddings import (
    build_quiz_clustering_texts,
    cluster_quiz_titles,
    generate_embeddings,
)
from core.database import find_similar_videos, video_exists, add_video_embeddings, get_video_count
from core.llm import get_llm_client

RECOMMENDATION_THRESHOLD = 0.8
MAX_RECOMMENDATIONS = 3
RECOMMENDATION_CANDIDATE_LIMIT = 10
MERGE_TIMESTAMP_GAP_SECONDS = 30


recommendation_routes = Blueprint('recommendations', __name__)
logger = logging.getLogger(__name__)


def _log_final_json(marker, payload):
    logger.info("%s %s", marker, json.dumps(payload, ensure_ascii=False, default=str))


def _merge_recommendation_chunks(chunks):
    chunks_by_video = {}
    for chunk in chunks:
        chunks_by_video.setdefault(chunk.get('video_id'), []).append(chunk)

    merged_ranges = []
    for video_id, video_chunks_for_video in chunks_by_video.items():
        sorted_chunks = sorted(video_chunks_for_video, key=lambda chunk: chunk.get('start', 0))
        ranges = []

        for chunk in sorted_chunks:
            if not ranges:
                ranges.append({
                    **chunk,
                    'text_parts': [chunk.get('text', '')],
                })
                continue

            current = ranges[-1]
            gap = chunk.get('start', 0) - current.get('end', 0)
            if gap <= MERGE_TIMESTAMP_GAP_SECONDS:
                current['end'] = max(current.get('end', 0), chunk.get('end', 0))
                current['score'] = max(current.get('score', 0), chunk.get('score', 0))
                text = chunk.get('text', '')
                if text and text not in current['text_parts']:
                    current['text_parts'].append(text)
            else:
                ranges.append({
                    **chunk,
                    'text_parts': [chunk.get('text', '')],
                })

        best_range = max(ranges, key=lambda item: item.get('score', 0))
        best_range['video_id'] = video_id
        best_range['text'] = ' '.join(part for part in best_range.pop('text_parts') if part)
        merged_ranges.append(best_range)

    return sorted(merged_ranges, key=lambda item: item.get('score', 0), reverse=True)[:MAX_RECOMMENDATIONS]


@recommendation_routes.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Smart video recommendation system
    - First tries local database
    - If not enough good results: automatically searches YouTube
    - Imports new videos on demand
    - Returns best matches
    """
    try:
        data = request.get_json()
        if not data or 'incorrect_questions' not in data:
            return jsonify({"error": "Missing 'incorrect_questions' array"}), 400
        
        incorrect_questions = data['incorrect_questions']
        all_recommendations = {}
        
        
        question_texts = []
        for q in incorrect_questions:
            embed_text = f"{q.get('question_text', '')}"
            if q.get('correct_answer'):
                embed_text += f"\nCorrect Answer: {q.get('correct_answer', '')}"
            if q.get('user_answer'):
                embed_text += f"\nUser answered incorrectly: {q.get('user_answer', '')}"
            question_texts.append(embed_text)
        
        embeddings = generate_embeddings(question_texts)
        
        for idx, question in enumerate(incorrect_questions):
            question_id = question.get('id')
            
            
            recommendations = find_similar_videos(
                embeddings[idx],
                limit=RECOMMENDATION_CANDIDATE_LIMIT,
                min_score=RECOMMENDATION_THRESHOLD,
            )
            recommendations = _merge_recommendation_chunks(recommendations)
            
            
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append({
                    'video_id': rec['video_id'],
                    'video_title': rec['video_title'],
                    'text': rec['text'],
                    'start_time': rec['start'],
                    'end_time': rec['end'],
                    'youtube_url': f"https://www.youtube.com/watch?v={rec['video_id']}&t={int(rec['start'])}",
                    'relevance_score': round(rec['score'], 4)
                })
            
            all_recommendations[question_id] = {
                'question_text': question.get('question_text', ''),
                'recommendations': formatted_recommendations
            }

        _log_final_json("FINAL_RECOMMENDATIONS_JSON", {
            "incorrect_question_count": len(incorrect_questions),
            "threshold": RECOMMENDATION_THRESHOLD,
            "max_recommendations": MAX_RECOMMENDATIONS,
            "candidate_limit": RECOMMENDATION_CANDIDATE_LIMIT,
            "input": incorrect_questions,
            "result": all_recommendations,
        })

        
        return jsonify(all_recommendations)
    
    except Exception as e:
        logger.exception("Error getting recommendations")
        return jsonify({"error": "Failed to get recommendations"}), 500


@recommendation_routes.route('/api/recommendations/library/add', methods=['POST'])
@login_required
def add_video_to_library():
    """Admin endpoint to add new YouTube video to recommendation library"""
    try:
        data = request.get_json()
        if not data or 'video_url' not in data or 'title' not in data:
            return jsonify({"error": "Missing 'video_url' or 'title'"}), 400
        
        from core.embeddings import add_youtube_video
        from core.embeddings import extract_youtube_video_id
        
        video_id = extract_youtube_video_id(data['video_url'])
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400
        
        chunks_added = add_youtube_video(video_id, data['title'])
        
        return jsonify({
            "message": f"Added {chunks_added} chunks for video",
            "chunks_added": chunks_added,
            "video_id": video_id
        })
    
    except Exception as e:
        logger.exception("Error adding video")
        return jsonify({"error": "Failed to add video"}), 500


@recommendation_routes.route('/api/recommendations/library/stats', methods=['GET'])
def get_library_stats():
    """Get statistics about video recommendation library"""
    try:
        stats = get_video_count()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": "Failed to get library stats"}), 500


# Global cluster cache per user
_cluster_cache = {}  # user_id -> { clusters: [], names: {}, hash: str }
_cluster_warmups = set()
_cluster_warmups_lock = threading.Lock()

def get_list_hash(items: list) -> str:
    """Generate stable hash for list comparison"""
    import hashlib
    return hashlib.sha256('\n'.join(items).encode()).hexdigest()


def _extract_cluster_inputs(data):
    quiz_items = data.get('quizzes')
    if isinstance(quiz_items, list) and quiz_items:
        titles = [
            str(item.get('title', '')).strip() or 'Untitled Quiz'
            for item in quiz_items
        ]
        clustering_texts = build_quiz_clustering_texts(quiz_items)
        return titles, clustering_texts

    titles = data.get('titles', [])
    if not isinstance(titles, list):
        titles = []
    return titles, build_quiz_clustering_texts(titles)


def _build_cluster_result(titles, clustering_texts):
    if len(titles) < 2:
        return {
            "status": "ready",
            "clusters": [0] * len(titles),
            "count": 1,
            "names": {0: "Quizzes"},
            "hash": get_list_hash(clustering_texts)
        }

    clusters = cluster_quiz_titles(clustering_texts)
    cluster_count = len(set(clusters))
    cluster_names = {}

    llm_client = get_llm_client()
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
                logger.warning("Error naming cluster %s: %s", cluster_id, e)

    return {
        "status": "ready",
        "clusters": clusters,
        "count": cluster_count,
        "names": cluster_names,
        "hash": get_list_hash(clustering_texts)
    }


def warm_cluster_cache_for_user(user_id):
    """Build cluster cache for a logged-in user in the background."""
    if not user_id:
        return

    cache_key = str(user_id)
    with _cluster_warmups_lock:
        if user_id in _cluster_cache or cache_key in _cluster_warmups:
            return
        _cluster_warmups.add(cache_key)

    def run_warmup():
        try:
            quizzes = list(get_db().quizzes.find({"userId": user_id}))
            titles = [
                str(quiz.get('title', '')).strip() or 'Untitled Quiz'
                for quiz in quizzes
            ]
            clustering_texts = build_quiz_clustering_texts(quizzes)

            result = _build_cluster_result(titles, clustering_texts)
            _cluster_cache[user_id] = result
            logger.info("Warmed cluster cache for user %s with %s quizzes", user_id, len(titles))
        except Exception:
            logger.exception("Cluster cache warmup failed for user %s", user_id)
        finally:
            with _cluster_warmups_lock:
                _cluster_warmups.discard(cache_key)

    threading.Thread(target=run_warmup, daemon=True).start()


@recommendation_routes.route('/api/cluster-quizzes/clusterize', methods=['POST'])
def cluster_quizzes_full():
    """
    FULL CLUSTERIZATION ENDPOINT
    Heavy operation: Generate embeddings, run KMeans, generate LLM names
    Run this ONLY when quiz list actually changes
    """
    try:
        data = request.get_json()
        titles, clustering_texts = _extract_cluster_inputs(data or {})
        
        user_id = get_current_user_db_id() or "guest"
        
        if len(titles) < 2:
            result = _build_cluster_result(titles, clustering_texts)
            _cluster_cache[user_id] = result
            _log_final_json("FINAL_CLUSTER_RESULT_JSON", {
                "user_id": user_id,
                "title_count": len(titles),
                "titles": titles,
                "result": result,
            })
            return jsonify(result)

        result = _build_cluster_result(titles, clustering_texts)
        
        # Cache this result permanently for this user
        _cluster_cache[user_id] = result
        _log_final_json("FINAL_CLUSTER_RESULT_JSON", {
            "user_id": user_id,
            "title_count": len(titles),
            "titles": titles,
            "result": result,
        })
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception("Error getting clusters")
        return jsonify({"error": "Failed to get clusters"}), 500


@recommendation_routes.route('/api/cluster-quizzes/extract', methods=['POST'])
def cluster_quizzes_extract():
    """
    EXTRACT ONLY ENDPOINT
    INSTANT OPERATION: Return already cached clusters from memory
    Run this when user toggles cluster switch
    Returns status=missing if no cache exists - client should call /clusterize then
    """
    try:
        data = request.get_json()
        _, clustering_texts = _extract_cluster_inputs(data or {})
        
        user_id = get_current_user_db_id() or "guest"
        
        current_hash = get_list_hash(clustering_texts)
        
        if user_id in _cluster_cache and _cluster_cache[user_id]['hash'] == current_hash:
            result = _cluster_cache[user_id]
            _log_final_json("FINAL_CLUSTER_CACHE_JSON", {
                "user_id": user_id,
                "result": result,
            })
            return jsonify(result)
        else:
            return jsonify({"status": "missing"})
            
    except Exception as e:
        logger.exception("Error extracting clusters")
        return jsonify({"error": "Failed to get clusters"}), 500
