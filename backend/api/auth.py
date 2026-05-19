import traceback
import datetime
from flask import Blueprint, jsonify, request
from flask_login import login_user, logout_user, login_required, current_user
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from bson import ObjectId

from config import GOOGLE_CLIENT_ID, User, get_db

auth_routes = Blueprint('auth', __name__)


@auth_routes.route('/api/auth/google/callback', methods=['POST'])
def google_callback():
    """Handles the token received from Google Sign-In on the frontend."""
    if not GOOGLE_CLIENT_ID:
         return jsonify({"error": "Google Sign-In not configured on server."}), 503

    data = request.get_json()
    token = data.get('credential')

    if not token:
        return jsonify({"error": "Missing credential token."}), 400 

    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        db = get_db()
        users_collection = db.users
        google_id = idinfo['sub']
        email = idinfo.get('email')
        name = idinfo.get('name')
        picture = idinfo.get('picture')

        if not email:
            return jsonify({"error": "Email not found in Google token."}), 400

        user_data = users_collection.find_one({"googleId": google_id})
        current_time = datetime.datetime.now(datetime.timezone.utc)

        if user_data:
            update_result = users_collection.update_one(
                {"_id": user_data['_id']},
                {"$set": {
                    "lastLogin": current_time,
                    "name": name,
                    "picture": picture,
                    "email": email
                }}
            )
            user_data = users_collection.find_one({"_id": user_data['_id']}) 
            print(f"Login: {email}")
        else:
            new_user_doc = {
                "googleId": google_id,
                "email": email,
                "name": name,
                "picture": picture,
                "createdAt": current_time,
                "lastLogin": current_time
            }
            insert_result = users_collection.insert_one(new_user_doc)
            user_data = new_user_doc
            user_data['_id'] = insert_result.inserted_id 
            print(f"New user: {email}")

        user_obj = User(user_data)
        login_user(user_obj, remember=True, duration=datetime.timedelta(days=30))

        from api.recommendations import warm_cluster_cache_for_user
        warm_cluster_cache_for_user(user_data['_id'])

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": str(user_data['_id']),
                "email": user_data.get('email'),
                "name": user_data.get('name'),
                "picture": user_data.get('picture')
             }
        }), 200

    except ValueError as e:
         print(f"Google Token Verification Error: {e}")
         traceback.print_exc()
         return jsonify({"error": "Invalid Google sign-in token."}), 401
    except Exception as e:
         print(f"Error during Google callback processing: {e}")
         traceback.print_exc()
         return jsonify({"error": "Server error during authentication."}), 500


@auth_routes.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    user_email = current_user.email
    logout_user() 
    print(f"Logout: {user_email}")
    
    return jsonify({"message": "Logout successful"}), 200


@auth_routes.route('/api/auth/status', methods=['GET'])
def auth_status():
    if current_user and current_user.is_authenticated:
        return jsonify({
            "isAuthenticated": True,
            "user": {
                "id": current_user.id,
                "email": current_user.email,
                "name": current_user.name,
                "picture": current_user.picture
             }
        })
    else:
         return jsonify({"isAuthenticated": False, "user": None})
