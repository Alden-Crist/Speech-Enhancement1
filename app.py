from flask import Flask, request, send_from_directory, jsonify
import os
import shutil
import soundfile as sf
from prediction_denoise import prediction
from werkzeug.utils import secure_filename

# Directories for storing uploads and outputs
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Ensure the directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Welcome to the Audio Denoising API! Use /upload to send an audio file."

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_DIR, filename)
        file.save(input_path)

        # Set parameters for the denoising function
        weights_path = 'path/to/weights/folder'  # Specify path to weights folder
        name_model = 'your_model_name'  # Specify model name
        audio_output_filename = f"denoised_{filename}"
        output_path = os.path.join(OUTPUT_DIR, audio_output_filename)

        # Define other parameters
        sample_rate = 8000
        min_duration = 1
        frame_length = 8064
        hop_length_frame = 8064
        n_fft = 255
        hop_length_fft = 63

        try:
            # Call the prediction function to denoise the audio
            prediction(
                weights_path,
                name_model,
                UPLOAD_DIR,
                OUTPUT_DIR,
                filename,
                audio_output_filename,
                sample_rate,
                min_duration,
                frame_length,
                hop_length_frame,
                n_fft,
                hop_length_fft
            )

            # Send the denoised file as response
            return send_from_directory(OUTPUT_DIR, audio_output_filename, as_attachment=True)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
