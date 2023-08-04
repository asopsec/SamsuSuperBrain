import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class FineTuneController:
    def __init__(self):
        self.SQLALCHEMY_DATABASE_URL = "sqlite:///./database/sqlite/fine_tuner.database"
        self.engine = create_engine(self.SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.Base = declarative_base()

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Base.metadata.create_all(bind=engine)  <-- Remove this line

