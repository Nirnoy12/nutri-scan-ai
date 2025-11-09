import os
import logging
from datetime import datetime, UTC
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from paddleocr import PaddleOCR
from flask_sqlalchemy import SQLAlchemy
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification, 
    AutoTokenizer, AutoModelForCausalLM
)
from huggingface_hub import InferenceClient
from openai import OpenAI
from config import Config

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG)

# --- App Setup ---
# Tell Flask to use the 'public' folder as the static folder
# and serve its contents from the root path
app = Flask(__name__, static_folder='public', static_url_path='')
app.config.from_object(Config)

# --- Database Setup ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "You must be logged in to access this page."

# --- AI Client Setup ---
# Check if HF_TOKEN is set
if not app.config['HF_TOKEN']:
    raise ValueError("HF_TOKEN is not set. Please add it to your .env file.")

# --- PaddleOCR Setup ---
try:
    # Suppress verbose PaddleOCR logs
    logging.getLogger('ppocr').setLevel(logging.ERROR) 
    
    # Replaced 'use_angle_cls' and removed invalid 'show_log'
    ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
    logging.info("PaddleOCR initialized successfully.")
except Exception as e:
    logging.error(f"PaddleOCR initialization FAILED: {repr(e)}")
    ocr_model = None

# Define local food models
FOOD_MODEL_LOCAL_PRIMARY = "prithivMLmods/Food-101-93M"
FOOD_MODEL_LOCAL_FALLBACK = "nateraw/food"
food_client = None 
logging.info("Food recognition will use dual local models.")

# Initialize OpenAI client for chat
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
try:
    chat_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=app.config['HF_TOKEN'],
    )
    logging.info("OpenAI client for chat initialized successfully.")
except Exception as e:
    logging.error(f"OpenAI client initialization FAILED: {repr(e)}")
    chat_client = None

# ==================================
# --- DATABASE MODELS ---
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

def extract_text(image_path):
    """Uses PaddleOCR to read text from an image."""
    if not ocr_model:
        logging.error("PaddleOCR model is not available.")
        return "Error: OCR model is not initialized."
    try:
        logging.debug(f"Processing image with PaddleOCR: {image_path}")
        
        # *** FIX: Changed from .ocr(cls=True) to .predict() ***
        result = ocr_model.predict(image_path)
        
        if not result or not result[0]:
            logging.warning("PaddleOCR returned no text.")
            return "OCR returned no text. Image may be unclear."

        lines = []
        for res_line in result[0]:
            # res_line format is [[box], (text, confidence)]
            text = res_line[1][0]
            lines.append(text)
        
        full_text = "\n".join(lines)
        logging.debug(f"PaddleOCR extracted text: {full_text[:100]}...")
        return full_text.strip()
        
    except Exception as e:
        logging.error(f"PaddleOCR Error: {repr(e)}")
        return f"Error: PaddleOCR failed. (Reason: {repr(e)})"


def _recognize_food_local(image_path, model_name):
    """
    Helper function to run a local food recognition model.
    Returns (list_of_labels, None) or (None, error_string)
    """
    try:
        logging.debug(f"Loading local food model: {model_name}")
        
        processor = AutoImageProcessor.from_pretrained(model_name, token=app.config['HF_TOKEN'])
        model = AutoModelForImageClassification.from_pretrained(model_name, token=app.config['HF_TOKEN'])
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probs = logits.softmax(dim=-1)
        
        top_5_indices = probs[0].topk(5).indices
        top_5_labels = [model.config.id2label[i.item()].replace('_', ' ').title() for i in top_5_indices]
        
        logging.debug(f"Local food recognition successful ({model_name}): {top_5_labels}")
        return top_5_labels, None # Return list of labels, no error
    except Exception as e:
        logging.error(f"Local Food Model Error ({model_name}): {repr(e)}")
        return None, f"Error: Could not identify food (Local). (Reason: {repr(e)})"


def recognize_food(image_path):
    """
    Uses two local food models to find the "best pick" identification.
    Returns a clean text string (e.g., "Hamburger") and an error string (if any).
    """
    logging.debug(f"Starting dual food recognition for: {image_path}")

    # 1. Run primary local model (prithivMLmods)
    labels_1, err_1 = _recognize_food_local(
        image_path, 
        FOOD_MODEL_LOCAL_PRIMARY 
    )
    
    # 2. Run secondary local model (nateraw)
    labels_2, err_2 = _recognize_food_local(
        image_path, 
        FOOD_MODEL_LOCAL_FALLBACK
    )

    # 3. Check for total failure
    if err_1 and err_2:
        logging.error(f"All food models failed. Model 1: {err_1}, Model 2: {err_2}")
        return None, f"All food models failed.\nModel 1: {err_1}\nModel 2: {err_2}"

    # 4. "Best Pick" Logic
    best_pick_text = ""
    if labels_1 and labels_2:
        # Find common items (intersection)
        common_items = list(set(labels_1) & set(labels_2))
        if common_items:
            logging.debug(f"Found common items: {common_items}")
            best_pick_text = ", ".join(common_items)
        else:
            # No common items, trust the primary model's top guess
            logging.debug(f"No common items. Trusting primary model: {labels_1[0]}")
            best_pick_text = labels_1[0]
    elif labels_1:
        # Only primary model succeeded
        logging.debug(f"Only primary model succeeded. Using: {labels_1[0]}")
        best_pick_text = labels_1[0]
    elif labels_2:
        # Only secondary model succeeded
        logging.debug(f"Only secondary model succeeded. Using: {labels_2[0]}")
        best_pick_text = labels_2[0]
    else:
        # This case should be caught by step 3, but as a safeguard
        return None, "Food models returned empty lists."

    logging.debug(f"Best pick for food: {best_pick_text}")
    return best_pick_text, None # Return clean string, no error

def get_ai_nutrition_analysis(context_text, system_prompt, user_prompt):
    """
    Calls the chat API to get a nutritional analysis.
    Returns (analysis_text, None) on success, or (None, error_string) on failure.
    """
    if not chat_client:
        logging.error("Chat client is not initialized.")
        return None, "Chat client is not available."

    try:
        logging.debug(f"Calling chat API with model: {CHAT_MODEL}")
        logging.debug(f"System Prompt: {system_prompt}")
        logging.debug(f"User Prompt: {user_prompt} | Context: {context_text[:100]}...")
        
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

def chat_with_bot_local(message):
    """Uses local DistilGPT-2 model for chatbot conversation."""
    CHAT_FALLBACK_MODEL = "distilgpt2"
    try:
        logging.debug(f"Loading local chat model: {CHAT_FALLBACK_MODEL}")
        chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_FALLBACK_MODEL, token=app.config['HF_TOKEN'])
        chat_model = AutoModelForCausalLM.from_pretrained(CHAT_FALLBACK_MODEL, token=app.config['HF_TOKEN'])
        user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(10).all()
        eaten_foods = [scan.quick_verdict for scan in user_scans]
        system_prompt = f"You are NutriBot, a helpful AI nutrition assistant. The user's recent scans: {', '.join(eaten_foods)}. Be concise and helpful."
        prompt = f"{system_prompt}\n\nUser: {message}"
        logging.debug(f"Chat prompt: {prompt}")
        inputs = chat_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = chat_model.generate(
            inputs["input_ids"],
            max_new_tokens=250,
            temperature=0.7,
            pad_token_id=chat_tokenizer.eos_token_id
        )
        response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        logging.debug(f"Chat response: {response}")
        return response
    except Exception as e:
        logging.error(f"Local Chat Model Error: {repr(e)}")
        return f"Error: Chatbot failed. (Reason: {repr(e)})"

# ==================================
# --- AUTHENTICATION ROUTES ---
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
            "filename": scan.filename,
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
            # 1. Get OCR text from PaddleOCR
            logging.debug("Scan type 'label': Starting PaddleOCR.")
            ocr_text_to_save = extract_text(filepath) # This now uses Paddle
            if "Error:" in ocr_text_to_save:
                logging.error(f"PaddleOCR failed: {ocr_text_to_save}")
                return jsonify({'error': ocr_text_to_save}), 400

            # 2. Get AI Quick Verdict
            verdict_sys_prompt = "You are a professional nutritionist. You have read the following nutrition label text."
            verdict_user_prompt = "Provide a concise, one-paragraph verdict on this product's healthiness based on the text. Speak as the nutritionist."
            verdict, err = get_ai_nutrition_analysis(ocr_text_to_save, verdict_sys_prompt, verdict_user_prompt)
            if err:
                logging.error(f"AI Verdict failed: {err}")
                return jsonify({'error': f"AI analysis failed: {err}"}), 500
            quick_verdict = verdict

            # 3. Get AI Detailed Report (Long-Term)
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
            # 1. Get "best pick" food recognition text
            logging.debug("Scan type 'food': Starting dual food recognition.")
            best_pick_text, err = recognize_food(filepath) 
            
            if err:
                logging.error(f"Food recognition failed: {err}")
                return jsonify({'error': err}), 400
            
            ocr_text_to_save = best_pick_text # This is the clean string, e.g., "Hamburger"

            # 2. Get AI Quick Verdict
            verdict_sys_prompt = "You are a professional nutritionist. A food item has been identified."
            verdict_user_prompt = f"The food is: {ocr_text_to_save}. Provide a concise, one-paragraph nutritional verdict on this item. Speak as the nutritionist."
            verdict, err = get_ai_nutrition_analysis(ocr_text_to_save, verdict_sys_prompt, verdict_user_prompt)
            if err:
                logging.error(f"AI Verdict failed: {err}")
                return jsonify({'error': f"AI analysis failed: {err}"}), 500
            quick_verdict = verdict
            
            # 3. Get AI Detailed Report (Long-Term)
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
            ocr_text=ocr_text_to_save, # Save the PaddleOCR text or "best pick" food
            user_id=current_user.id
        )
        db.session.add(new_scan)
        db.session.commit()
        logging.debug(f"Scan saved to DB: {filename}, type: {scan_type}")

        return jsonify({
            "timestamp": new_scan.timestamp.strftime("%Y-%m-%d %H:%M"),
            "filename": new_scan.filename,
            "ocr_text": new_scan.ocr_text,
            "quick_verdict": new_scan.quick_verdict,
            "detailed_report": detailed_report, 
            "type": new_scan.scan_type
        })
    except Exception as e:
        logging.error(f"Analyze Error: {repr(e)}")
        return jsonify({'error': f'An unexpected error occurred: {repr(e)}'}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat_with_bot():
    """Handles chatbot conversation using HF Inference API with local fallback."""
    logging.debug(f"Received /chat request: {request.json}")
    
    if not request.json or 'message' not in request.json:
        logging.error("No message provided in request")
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = request.json['message'].strip()
    if not user_message:
        logging.error("Message cannot be empty")
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    try:
        user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(10).all()
        eaten_foods = [scan.quick_verdict for scan in user_scans] 
        system_prompt = f"You are NutriBot, a helpful AI nutrition assistant. The user's recent scan history (verdicts): {', '.join(eaten_foods)}. Be concise and helpful."
        
        # Try OpenAI-compatible Inference API
        if chat_client:
            try:
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
        
        # Fallback to local DistilGPT-2
        response = chat_with_bot_local(user_message)
        if "Error:" in response:
            logging.error(f"Chat fallback failed: {response}")
            return jsonify({'error': response}), 500
        logging.debug(f"Chat fallback response: {response}")
        return jsonify({'reply': response})
    except Exception as e:
        logging.error(f"Chat Error: {repr(e)}")
        return jsonify({'error': f"Chatbot error: {repr(e)}"}), 500

# ==================================
# --- APP INITIALIZATION ---
# ==================================
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
    # Run Flask on all network interfaces, port 8080 (changeable)
    app.run(host='0.0.0.0', port=8080, debug=True)
