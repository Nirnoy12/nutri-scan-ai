import os
from dotenv import load_dotenv

# Find the .env file
basedir = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(basedir, '.env')

# --- DEBUG CODE ---
if not os.path.exists(env_path):
    print("--- FATAL ERROR: .env file NOT FOUND! ---")
    print(f"--- Looking for it at: {env_path} ---")
else:
    print(f"--- SUCCESS: .env file found at {env_path} ---")
    load_dotenv(env_path)
# --------------------

class Config:
    """Set Flask configuration variables from .env file."""
    
    # General Config
    SECRET_KEY = os.environ.get('SECRET_KEY')
    FLASK_APP = 'app.py'

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Hugging Face
    # --- THIS IS THE FIX ---
    # It will now be 'None' if the key is not in the .env file
    HF_TOKEN = os.environ.get('HF_TOKEN') 

    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    # App-specific
    UPLOAD_FOLDER = 'public/uploads'