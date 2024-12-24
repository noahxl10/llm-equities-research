from datetime import timedelta
import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    TESTING = False
    UPLOAD_FOLDER = 'uploads'
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=10)
    API_KEY = os.environ['FLASK_API_KEY']
    SECRET_KEY = os.environ['FLASK_SECRET_KEY']
    REFRESH_SECRET_KEY = os.environ['FLASK_REFRESH_SECRET_KEY']
    ALLOWED_IPS = ['50.201.186.54', '216.194.126.180', '127.0.0.1']
