import os
from dotenv import load_dotenv

# Find the .env file
basedir = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(basedir, '.env')
load_dotenv(env_path)

# --- DATABASE SETUP ---
# Get the production database URL from the environment
DATABASE_URL = os.environ.get('DATABASE_URL')

class Config:
    """Set Flask configuration variables from .env file."""
    
    # General Config
    SECRET_KEY = os.environ.get('SECRET_KEY')
    FLASK_APP = 'app.py'

    # --- THIS IS THE CRITICAL DATABASE FIX ---
    if DATABASE_URL:
        # We are on Render (production)
        # Fix for SQLAlchemy: 'postgres://' needs to be 'postgresql://'
        SQLALCHEMY_DATABASE_URI = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    else:
        # We are local
        # Use a local SQLite database file named 'app.db'
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(basedir, 'app.db')}"
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- AI & Other Keys ---
    HF_TOKEN = os.environ.get('HF_TOKEN') 
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    # --- This path MUST match your Render Disk Mount Path ---
    UPLOAD_FOLDER = 'public/uploads'