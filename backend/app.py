from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")  # Set the backend to 'Agg' (Non-GUI)
import matplotlib.pyplot as plt
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
from flask_sqlalchemy import SQLAlchemy
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# Configure JWT
app.config["JWT_SECRET_KEY"] = "your_secret_key"
jwt = JWTManager(app)

# Configure SQLite Database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    audio_files = db.relationship('AudioFile', backref='user', lazy=True)

class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    audio_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    mel_spectrogram_image = db.Column(db.String(255), nullable=False)
    prediction_image = db.Column(db.String(255), nullable=False)

# Create tables
with app.app_context():
    db.create_all()

# Register Route
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        access_token = create_access_token(identity=user.id)  # Use user.id instead of username
        return jsonify({"token": access_token, "user_id": user.id}), 200  # Include user_id

    return jsonify({"error": "Invalid credentials"}), 401

# Add these constants near other constants
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_9967_mel.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, "images")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

audios=['Fire', 'Rain', 'Thunderstorm', 'WaterDrops', 'Wind', 'Silence', 'TreeFalling', 'Helicopter', 'VehicleEngine', 'Axe', 'Chainsaw', 'Generator', 'Handsaw', 'Firework', 'Gunshot', 'WoodChop', 'Whistling', 'Speaking', 'Footsteps', 'Clapping', 'Insect', 'Frog', 'BirdChirping', 'WingFlaping', 'Lion', 'WolfHowl', 'Squirrel']

def apply_noise_reduction(data, rate):
    return nr.reduce_noise(y=data, sr=rate)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def normalize_audio(data):
    max_val = np.max(np.abs(data))
    return data / max_val if max_val != 0 else data

def extract_mfcc_and_save_image(file_path, image_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title(f"MFCC - {os.path.basename(file_path)}")
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def prediction(image_path):
    try:
        # Convert to absolute path
        abs_image_path = os.path.abspath(image_path)
        
        print(f"Debug - Image path: {abs_image_path}")
        print(f"Debug - Model path: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at: {MODEL_PATH}")
            return "Error: Model file not found"
            
        if not os.path.exists(abs_image_path):
            print(f"Error: Image file not found at: {abs_image_path}")
            return "Error: Image file not found"

        print("Debug - Loading image...")
        # Load image and convert to array
        img = image.load_img(abs_image_path, target_size=(224, 224, 3))
        img_array = image.img_to_array(img)

        # Convert to numpy array and normalize
        X = np.expand_dims(img_array, axis=0)
        X = X / 255.0

        print("Debug - Loading model...")
        # Load the model
        try:
            model = load_model(MODEL_PATH)
            print("Debug - Model loaded successfully")
        except Exception as model_error:
            print(f"Error loading model: {str(model_error)}")
            return f"Error: Failed to load model - {str(model_error)}"

        print("Debug - Making prediction...")
        # Predict
        predictions = model.predict(X)
        predicted_class = np.argmax(predictions)

        print(f"Debug - Prediction complete. Class: {predicted_class}")
        return audios[int(predicted_class)]

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Python version: {sys.version}")
        print(f"Current working directory: {os.getcwd()}")
        return f"Error: {str(e)}"

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        token = request.headers.get("Authorization")
        user_id = request.form.get("user_id")
        
        print(f"Debug - Processing upload for user: {user_id}")

        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"Debug - File saved to: {file_path}")
        
        rate, data = wavfile.read(file_path)
        
        if data.ndim == 2:
            left_channel = apply_noise_reduction(data[:, 0], rate)
            right_channel = apply_noise_reduction(data[:, 1], rate)
            reduced_noise = np.column_stack((left_channel, right_channel))
        else:
            reduced_noise = apply_noise_reduction(data, rate)
        
        cutoff_frequency = 4000  
        filtered_audio = butter_lowpass_filter(reduced_noise, cutoff_frequency, rate)
        normalized_audio = normalize_audio(filtered_audio)
        
        processed_file_path = os.path.join(OUTPUT_FOLDER, file.filename)
        wavfile.write(processed_file_path, rate, np.int16(normalized_audio * 32767))
        
        # Generate spectrogram image with a unique filename
        spectrogram_filename = f"{os.path.splitext(file.filename)[0]}.png"
        image_path = os.path.join(OUTPUT_FOLDER, spectrogram_filename)
        extract_mfcc_and_save_image(processed_file_path, image_path)
        print(f"Debug - Spectrogram saved to: {image_path}")
        
        prediction_result = prediction(image_path)
        print(f"Debug - Prediction result: {prediction_result}")
        
        # Get prediction result
        image_filename = f"{prediction_result}.jpg"
        image_path1 = os.path.join(IMAGES_FOLDER, image_filename)
        
        # Create relative URLs for frontend
        spectrogram_url = f"/processed/{spectrogram_filename}"
        prediction_image_url = f"/static/images/{image_filename}"
        
        try:
            new_audio = AudioFile(
                user_id=user_id,
                audio_path=file_path,
                prediction=prediction_result,
                mel_spectrogram_image=image_path,
                prediction_image=image_path1
            )
            db.session.add(new_audio)
            db.session.commit()
            print("Debug - Database entry created successfully")
        except Exception as db_error:
            print(f"Error saving to database: {str(db_error)}")
            return jsonify({"error": f"Database error: {str(db_error)}"}), 500

        return jsonify({
            "message": "File processed successfully",
            "image_path": spectrogram_url,  # Return the URL instead of file path
            "prediction": prediction_result,
            "prediction_image": prediction_image_url
        }), 200
    except Exception as e:
        print(f"Error in upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_image", methods=["GET"])
def get_image():
    try:
        image_path = request.args.get("image_path")
        if not image_path:
            return jsonify({"error": "No image path provided"}), 400
            
        # Handle the new URL format
        if image_path.startswith('/processed/'):
            filename = os.path.basename(image_path)
            return send_from_directory(OUTPUT_FOLDER, filename)
        elif image_path.startswith('/static/images/'):
            filename = os.path.basename(image_path)
            return send_from_directory(IMAGES_FOLDER, filename)
        else:
            # For backward compatibility, try to serve from absolute path
            abs_image_path = os.path.abspath(image_path)
            if not os.path.exists(abs_image_path):
                return jsonify({"error": f"Image not found at: {abs_image_path}"}), 404
            
            directory = os.path.dirname(abs_image_path)
            filename = os.path.basename(abs_image_path)
            return send_from_directory(directory, filename)
            
    except Exception as e:
        print(f"Error in get_image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/processed/<filename>')
def serve_processed_image(filename):
    try:
        return send_from_directory(OUTPUT_FOLDER, filename)
    except Exception as e:
        print(f"Error serving processed image: {str(e)}")
        return jsonify({"error": str(e)}), 404

@app.route('/static/images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory(IMAGES_FOLDER, filename)
    except Exception as e:
        print(f"Error serving static image: {str(e)}")
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    app.run(debug=True)