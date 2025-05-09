# WildGuard AI 

An intelligent audio classification system designed to detect and classify various sounds in wildlife environments, helping in forest monitoring and conservation efforts.

## 🎯 Features

- **Real-time Audio Classification**: Identifies 27 different types of sounds including:
  - Natural Sounds: Fire, Rain, Thunderstorm, Water Drops, Wind
  - Wildlife Sounds: Bird Chirping, Frog, Insect, Lion, Wolf Howl, Squirrel
  - Human Activity: Footsteps, Speaking, Whistling, Clapping
  - Machinery: Helicopter, Vehicle Engine, Generator
  - Forest Activities: Tree Falling, Axe, Chainsaw, Handsaw, Wood Chop
  - Other: Silence, Firework, Gunshot

- **Audio Processing**:
  - Noise reduction and filtering
  - Audio normalization
  - Mel-spectrogram generation
  - MFCC (Mel-frequency cepstral coefficients) analysis

- **User Authentication**:
  - Secure user registration and login
  - JWT-based authentication
  - Password hashing

- **Database Integration**:
  - SQLite database for storing user data
  - Audio file management
  - Prediction history tracking

## 🛠️ Technology Stack

- **Backend**:
  - Flask (Python web framework)
  - TensorFlow/Keras (Deep Learning)
  - Librosa (Audio processing)
  - SQLAlchemy (ORM)
  - JWT (Authentication)

- **Audio Processing Libraries**:
  - scipy
  - noisereduce
  - numpy
  - matplotlib
  - soundfile

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- virtual environment

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhavna3/wildguardai.git
   cd wildguardai
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**
   ```bash
   # The database will be automatically created when you run the application
   ```

5. **Run the application**
   ```bash
   cd backend
   python app.py
   ```

## 📁 Project Structure

```
WildGuard-AI/
├── backend/
|   ├── database.py 
│   ├── app.py                 # Main Flask application
│   ├── model_9967_mel.h5      # Trained ML model
│   ├── static/
│   │   └── images/            # Prediction images
│   ├── uploads/               # Temporary audio file storage
│   ├── processed/             # Processed audio files and spectrograms
│   └──instance/
├── frontend/
|   ├──  index.html
│   ├── src/
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## 🎵 Supported Audio Formats

- WAV files
- Recommended sampling rate: 44.1kHz
- Supports both mono and stereo audio

## 🔒 API Endpoints

### Authentication
- `POST /register` - Register new user
- `POST /login` - User login

### Audio Processing
- `POST /upload` - Upload and process audio file
- `GET /processed/<filename>` - Get processed spectrogram
- `GET /static/images/<filename>` - Get prediction images

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Project Link: [https://github.com/bhavna3/wildguardai](https://github.com/bhavna3/wildguardai)
