from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import os


class Config:
    db_host = os.environ["DB_HOST"]
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_USER_PASSWORD"]
    db_port = os.environ["DB_PORT"]
    db_database = os.environ["DB_DATABASE"]
    SQLALCHEMY_DATABASE_URI = DATABASE_URL = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False


Engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
Session = scoped_session(sessionmaker(bind=Engine))
db_session = Session
