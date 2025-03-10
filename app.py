"""from flask import Flask, request, jsonify, redirect, url_for
import real_time_video  # Import your model here

app = Flask(__name__)

# Route for emotion detection
@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()

    # Here, you would run your emotion detection model on the input
    # For example, emotion = run_emotion_detection(data['input'])
    # Assuming the model detects the emotion as "happy" for now.
    
    emotion = real_time_video.detect_emotion(data)  # This function uses your model
    
    # Return the detected emotion
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True) """
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from real_time_video import detect_emotion_from_video  # Import your model function

app = Flask(__name__)
CORS(app)

# Main route for the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route for detecting emotion
@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if request.method == 'POST':
        try:
            emotion = detect_emotion_from_video()  # Assuming this function captures video and returns emotion
            return jsonify({'emotion': emotion}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    app.run(debug=True)

    
