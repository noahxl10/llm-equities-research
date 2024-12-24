from flask import Blueprint, request, jsonify, make_response
from flask_jwt_extended import JWTManager
import datetime
import traceback
import jwt

from db.db_session import db_session
from db.data_models import AppUser
from services.utils import now
from config.config import Config

auth_blueprint = Blueprint('auth_blueprint', __name__)

@auth_blueprint.route("/login", methods=["POST"])
def login():
    """
    Basic login endpoint that issues JWT tokens.
    Replace with secure password hashing and verification as needed.
    """
    auth = request.authorization
    if not auth or not auth.username or not auth.password:
        return make_response("Could not verify", 401)

    # Example password check (very insecure, placeholder only)
    user = (
        db_session()
        .query(AppUser)
        .filter_by(username=auth.username, password=auth.password)
        .first()
    )
    if not user:
        return make_response("Could not verify", 401)

    try:
        access_token = jwt.encode(
            {"user": auth.username, "exp": now() + datetime.timedelta(minutes=60)},
            Config.SECRET_KEY,
            algorithm="HS256",
        )
        refresh_token = jwt.encode(
            {"user": auth.username, "exp": now() + datetime.timedelta(days=30)},
            Config.JWT_REFRESH_SECRET_KEY,
            algorithm="HS256",
        )
        return jsonify({"access_token": access_token, "refresh_token": refresh_token})
    except Exception as e:
        traceback.print_exc()
        return make_response("Token generation failed", 500)

@auth_blueprint.route("/token/refresh", methods=["POST"])
def refresh_token():
    """
    Refreshes the user's access token with the provided refresh token.
    """
    refresh_token = request.json.get("refresh_token")
    if not refresh_token:
        return jsonify({"message": "Refresh token is missing!"}), 401

    try:
        data = jwt.decode(
            refresh_token, Config.JWT_REFRESH_SECRET_KEY, algorithms=["HS256"]
        )
        new_access_token = jwt.encode(
            {"user": data["user"], "exp": now() + datetime.timedelta(minutes=60)},
            Config.SECRET_KEY,
            algorithm="HS256",
        )
        return jsonify({"access_token": new_access_token})
    except Exception:
        return jsonify({"message": "Refresh token is invalid!"}), 401