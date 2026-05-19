import os
import logging
from flask import Flask
from flask_login import current_user
from config import init_app, IS_PRODUCTION

from api.auth import auth_routes
from api.quizzes import quiz_routes
from api.chat import chat_routes
from api.recommendations import recommendation_routes


logger = logging.getLogger(__name__)


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
for noisy_logger in (
    "groq",
    "httpx",
    "httpcore",
    "urllib3",
    "werkzeug",
    "pymongo",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


app = Flask(__name__)
init_app(app)

app.register_blueprint(auth_routes)
app.register_blueprint(quiz_routes)
app.register_blueprint(chat_routes)
app.register_blueprint(recommendation_routes)


@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    return "OK", 200


@app.before_request
def before_request_handler():
    if current_user and current_user.is_authenticated:
        from api.recommendations import warm_cluster_cache_for_user
        warm_cluster_cache_for_user(current_user.get_db_id())


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = not IS_PRODUCTION
    app.run(debug=debug_mode, host='0.0.0.0', port=port, use_reloader=True)
