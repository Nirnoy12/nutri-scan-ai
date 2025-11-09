import os
import logging
import base64  # <-- ADDED
from datetime import datetime, UTC
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from huggingface_hub import InferenceClient
from openai import OpenAI
from config import Config
import io

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG)

# --- App Setup ---
app = Flask(__name__, static_folder='public', static_url_path='')
app.config.from_object(Config)

# --- Database Setup ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "You must be logged in to access this page."

# --- AI Client Setup ---

# Check for Hugging Face Token (for chat, OCR, and vision)
if not app.config['HF_TOKEN']:
    raise ValueError("HF_TOKEN is not set. Please add it to your .env file.")

# --- REMOVED: Google Gemini/Vision API Key Check ---

# --- PaddleOCR Setup ---
# *** ALL PADDLEOCR CODE REMOVED ***
logging.info("PaddleOCR has been removed. Using Hugging Face API.")

# --- Local Food Models ---
# *** ALL LOCAL TRANSFORMERS CODE REMOVED ***
logging.info("Local food models have been removed. Using Hugging Face API.")


# Initialize OpenAI client for chat, OCR, AND vision (via Hugging Face)
# This ONE client will handle all three tasks
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
OCR_MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita"
VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita" # Using LLaVA for food recognition

try:
    chat_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=app.config['HF_TOKEN'],
    )
    logging.info("OpenAI client for HF router initialized successfully.")
except Exception as e:
    logging.error(f"OpenAI client initialization FAILED: {repr(e)}")
    chat_client = None

# --- REMOVED: Google Cloud Vision Client ---

# ==================================
# --- DATABASE MODELS ---
# (This section is unchanged)
# ==================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(UTC))
    scan_type = db.Column(db.String(50), nullable=False)
    quick_verdict = db.Column(db.String(1000), nullable=False) # Increased size for AI verdicts
    ocr_text = db.Column(db.Text, nullable=True) # Will store combined OCR or food list
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ==================================
# --- AI HELPER FUNCTIONS ---
# ==================================

# --- NEW: Helper to convert local image to base64 data ---
def _image_to_base64_data_uri(image_path):
    """Converts an image file to a base64 data URI."""
    try:
        # Use PIL to open and convert to JPEG, ensuring compatibility
        img = Image.open(image_path)
        img = img.convert("RGB") # Ensure it's in RGB format
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        
        binary_data = buffered.getvalue()
        base64_encoded_data = base64.b64encode(binary_data)
        base64_string = base64_encoded_data.decode('utf-8')
        
        return f"data:image/jpeg;base64,{base64_string}"
    except Exception as e:
        logging.error(f"Failed to convert image to base64: {repr(e)}")
        raise

# --- REPLACED: extract_text ---
def extract_text(image_path):
    """Uses Hugging Face DeepSeek-OCR API to read text from an image."""
    if not chat_client:
        logging.error("HF client (chat_client) is not available.")
        return "Error: API is not initialized."

    try:
        logging.debug(f"Opening image for HF OCR: {image_path}")
        base64_image_uri = _image_to_base64_data_uri(image_path)
        
        prompt_text = "Extract all text from this image. Respond with only the transcribed text."

        logging.debug(f"Sending request to HF OCR model: {OCR_MODEL}")
        completion = chat_client.chat.completions.create(
            model=OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": base64_image_uri}}
                    ],
                }
            ],
            max_tokens=1024, # Give it enough tokens to read a full label
        )

        response_text = completion.choices[0].message.content.strip()
        
        logging.debug("HF OCR API success.")
        return response_text

    except Exception as e:
        logging.error(f"HF OCR API Error: {repr(e)}")
        return f"Error: OCR API failed. (Reason: {repr(e)})"


# --- REPLACED: recognize_food ---
def recognize_food(image_path):
    """Uses Hugging Face LLaVA API to identify food in an image."""
    if not chat_client:
        logging.error("HF client (chat_client) is not available.")
        return None, "Error: API is not initialized."

    try:
        logging.debug(f"Opening image for HF Vision: {image_path}")
        base64_image_uri = _image_to_base64_data_uri(image_path)
        
        prompt_text = "Identify the food item in this image. Respond with only the name of the food (e.g., 'Hamburger', 'Caesar Salad')."
        
        logging.debug(f"Sending request to HF Vision model: {VISION_MODEL}")
        completion = chat_client.chat.completions.create(
            model=VISION_MODEL, 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": base64_image_uri}}
                    ],
                }
            ],
            max_tokens=50, # We only need the food name
            temperature=0.1,
        )

        response_text = completion.choices[0].message.content.strip()
        
        # Clean the response just in case
        # (e.g., if it says "The food is: Hamburger", pull out "Hamburger")
        if ":" in response_text:
            response_text = response_text.split(":")[-1].strip().replace("\"", "").replace(".", "")
        
        # Sometimes models add "A picture of a..."
        response_text = response_text.split("A picture of a")[-1].strip()

        logging.debug(f"HF Vision API success. Food: {response_text}")
        return response_text, None

    except Exception as e:
        logging.error(f"HF Vision API Error: {repr(e)}")
        return None, f"Error: Vision API (HF) failed. (Reason: {repr(e)})"


def get_ai_nutrition_analysis(context_text, system_prompt, user_prompt):
    """
    Calls the chat API to get a nutritional analysis.
    (This function is unchanged)
    """
    if not chat_client:
        logging.error("Chat client is not initialized.")
        return None, "Chat client is not available."

    try:
        logging.debug(f"Calling chat API with model: {CHAT_MODEL}")
        full_user_content = f"{user_prompt}\n\nHere is the text to analyze:\n{context_text}"

        completion = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_content}
            ],
            max_tokens=350, 
            temperature=0.5, 
        )
        response = completion.choices[0].message.content
        logging.debug(f"Chat API analysis response: {response}")
        if response and response.strip():
            return response.strip(), None
        else:
            logging.warning("Chat API returned an empty analysis.")
            return None, "AI analysis returned an empty response."
    except Exception as e:
        logging.error(f"Chat API analysis Error: {repr(e)}")
        return None, f"Error during AI analysis: {repr(e)}"


# ==================================
# --- AUTHENTICATION ROUTES ---
# (This section is unchanged)
# ==================================
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash("Username already exists.")
        else:
            new_user = User(username=request.form['username'])
            new_user.set_password(request.form['password'])
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ==================================
# --- PROTECTED APP ROUTES ---
# (The /analyze route is unchanged, as the helper functions
#  have the same inputs/outputs as before)
# ==================================
@app.route('/')
@login_required
def home():
    return render_template('index.html', username=current_user.username)

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    """Fetches personalized history from the database."""
    try:
        user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(30).all()
        history_list = [{
            "filename": f"/uploads/{scan.filename}",
            "timestamp": scan.timestamp.strftime("%Y-%m-%d %H:%M"),
            "quick_verdict": scan.quick_verdict,
            "ocr_text": scan.ocr_text
        } for scan in user_scans]
        logging.debug(f"History fetched for user {current_user.id}: {len(history_list)} scans")
        return jsonify(history_list)
    except Exception as e:
        logging.error(f"History Error: {repr(e)}")
        return jsonify({'error': f'Failed to fetch history: {repr(e)}'}), 500

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_image():
    """Analyzes an image (OCR or Food) and saves to DB."""
    logging.debug(f"Received /analyze request: {request.form}, files: {request.files}")
    
    # ... (File validation checks are unchanged) ...
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    scan_type = request.form.get('scan_type', 'label')

    if file.filename == '':
        logging.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        logging.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': f"Invalid file type. Please upload a (png, jpg, jpeg, webp) file. Got: {file.filename}"}), 400
    
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    if file_size > 10 * 1024 * 1024:  # 10MB
        logging.error(f"File too large: {file_size} bytes")
        return jsonify({'error': f'File too large (Max 10MB). Got: {file_size} bytes'}), 400
    
    if scan_type not in ['label', 'food']:
        logging.error(f"Invalid scan_type: {scan_type}")
        return jsonify({'error': f"Invalid scan_type. Must be 'label' or 'food'. Got: {scan_type}"}), 400

    original_filename = secure_filename(file.filename)
    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{current_user.id}_{timestamp_str}_{original_filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        logging.debug(f"File saved: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save file: {repr(e)}")
        return jsonify({'error': f'Failed to save file: {repr(e)}'}), 500

    ocr_text_to_save = ""
    quick_verdict = ""
    detailed_report = []

    try:
        if scan_type == 'label':
            # 1. Get OCR text from Hugging Face API
            logging.debug("Scan type 'label': Starting HF DeepSeek-OCR.")
            ocr_text_to_save = extract_text(filepath) # This now uses HF
            if "Error:" in ocr_text_to_save:
                logging.error(f"HF OCR failed: {ocr_text_to_save}")
                return jsonify({'error': ocr_text_to_save}), 400

            # 2. Get AI Quick Verdict (Unchanged)
            verdict_sys_prompt = "You are a professional nutritionist. You have read the following nutrition label text."
            verdict_user_prompt = "Provide a concise, one-paragraph verdict on this product's healthiness based on the text. Speak as the nutritionist."
            verdict, err = get_ai_nutrition_analysis(ocr_text_to_save, verdict_sys_prompt, verdict_user_prompt)
            if err:
                logging.error(f"AI Verdict failed: {err}")
                return jsonify({'error': f"AI analysis failed: {err}"}), 500
            quick_verdict = verdict

            # 3. Get AI Detailed Report (Unchanged)
            report_sys_prompt = "You are a professional nutritionist."
            report_user_prompt = "Based on the nutrition label text, what are the potential long-term health impacts (positive or negative) of consuming this item regularly? Be concise and use bullet points."
            report, err = get_ai_nutrition_analysis(ocr_text_to_save, report_sys_prompt, report_user_prompt)
            if err:
                logging.error(f"AI Report failed: {err}")
                detailed_report = [{"nutrient": "Long-Term Impact", "impact": f"Failed to generate report: {err}"}]
            else:
                detailed_report = [{"nutrient": "Long-Term Impact", "impact": report}]
            
            logging.debug(f"Label analysis complete. Verdict: {quick_verdict[:50]}...")
            
        elif scan_type == 'food':
            # 1. Get food recognition text from Hugging Face API
            logging.debug("Scan type 'food': Starting HF LLaVA food recognition.")
            best_pick_text, err = recognize_food(filepath) 
            
            if err:
                logging.error(f"Food recognition failed: {err}")
                return jsonify({'error': err}), 400
            
            ocr_text_to_save = best_pick_text # This is the clean string, e.g., "Hamburger"

            # 2. Get AI Quick Verdict (Unchanged)
            verdict_sys_prompt = "You are a professional nutritionist. A food item has been identified."
            verdict_user_prompt = f"The food is: {ocr_text_to_save}. Provide a concise, one-paragraph nutritional verdict on this item. Speak as the nutritionist."
            verdict, err = get_ai_nutrition_analysis(ocr_text_to_save, verdict_sys_prompt, verdict_user_prompt)
            if err:
                logging.error(f"AI Verdict failed: {err}")
                return jsonify({'error': f"AI analysis failed: {err}"}), 500
            quick_verdict = verdict
            
            # 3. Get AI Detailed Report (Unchanged)
            report_sys_prompt = "You are a professional nutritionist."
            report_user_prompt = f"The food is: {ocr_text_to_save}. What are the potential long-term health impacts (positive or negative) of consuming this item regularly? Be concise and use bullet points."
            report, err = get_ai_nutrition_analysis(ocr_text_to_save, report_sys_prompt, report_user_prompt)
            if err:
                logging.error(f"AI Report failed: {err}")
                detailed_report = [{"nutrient": "Long-Term Impact", "impact": f"Failed to generate report: {err}"}]
            else:
                detailed_report = [{"nutrient": "Long-Term Impact", "impact": report}]
                
            logging.debug(f"Food analysis complete. Verdict: {quick_verdict[:50]}...")

        # --- Save to DB and return (This part is the same as before) ---
        new_scan = Scan(
            filename=filename,
            scan_type=scan_type,
            quick_verdict=quick_verdict,
            ocr_text=ocr_text_to_save, 
            user_id=current_user.id
        )
        db.session.add(new_scan)
        db.session.commit()
        logging.debug(f"Scan saved to DB: {filename}, type: {scan_type}")

        return jsonify({
            "timestamp": new_scan.timestamp.strftime("%Y-%m-%d %H:%M"),
            "filename": f"/uploads/{new_scan.filename}",
            "ocr_text": new_scan.ocr_text,
            "quick_verdict": new_scan.quick_verdict,
            "detailed_report": detailed_report, 
            "type": new_scan.scan_type
        })
    except Exception as e:
        db.session.rollback()
        logging.error(f"Analyze Error: {repr(e)}")
        try:
            os.remove(filepath)
            logging.debug(f"Removed failed upload: {filepath}")
        except OSError:
            pass
        return jsonify({'error': f'An unexpected error occurred: {repr(e)}'}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat_with_bot():
    """Handles chatbot conversation using HF Inference API."""
    # (This section is unchanged)
    logging.debug(f"Received /chat request: {request.json}")
    
    if not request.json or 'message' not in request.json:
        logging.error("No message provided in request")
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = request.json['message'].strip()
    if not user_message:
        logging.error("Message cannot be empty")
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    if not chat_client:
        logging.error("Chat client is not initialized.")
        return jsonify({'error': 'Chatbot is not configured.'}), 500

    try:
        user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(10).all()
        eaten_foods = [scan.quick_verdict for scan in user_scans] 
        system_prompt = f"You are NutriBot, a helpful AI nutrition assistant. The user's recent scan history (verdicts): {', '.join(eaten_foods)}. Be concise and helpful."
        
        logging.debug(f"Trying Inference API with chat model: {CHAT_MODEL}")
        completion = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=250,
            temperature=0.7,
        )
        response = completion.choices[0].message.content
        logging.debug(f"Chat API response: {response}")
        return jsonify({'reply': response})

    except Exception as e:
        logging.error(f"HF Chat API Error: {repr(e)}")
        return jsonify({'error': f"Chatbot failed: {repr(e)}"}), 500

# ==================================
# --- APP INITIALIZATION ---
# ==================================
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8080, debug=True)