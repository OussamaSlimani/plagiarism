import json
import os
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Constants
FEATURES_FILE = "features.json"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Flask Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained VGG16 model
model = VGG16(weights="imagenet", include_top=False)

def extract_features(image_path):
    """Extract deep learning features from an image."""
    image = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    image_array = img_to_array(image)  # Convert to NumPy array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocess for VGG16
    
    # Extract features using the CNN model
    features = model.predict(image_array)
    features = features.flatten()  # Flatten the feature map
    features = features / np.linalg.norm(features)  # Normalize the features
    return features

def load_features(file_path):
    """Load features from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return [np.array(feature) for feature in data]
        except json.JSONDecodeError:
            return []
    return []

def save_features(file_path, features):
    """Save features to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump([feature.tolist() for feature in features], file, indent=4)

def find_similarity(new_feature, stored_features):
    """Compare new feature with stored features."""
    for stored_feature in stored_features:
        similarity = cosine_similarity([new_feature], [stored_feature])[0][0]
        if similarity >= 0.9:
            return True
    return False

def allowed_file(filename):
    """Check if the uploaded file is allowed (image format)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    """API endpoint to check plagiarism."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Extract features from the uploaded image
            new_feature = extract_features(file_path)

            # Load stored features
            stored_features = load_features(FEATURES_FILE)

            # Check for similarity
            if find_similarity(new_feature, stored_features):
                return jsonify({"message": "Plagiarism detected."}), 200
            else:
                # Add new feature if not similar
                stored_features.append(new_feature)
                save_features(FEATURES_FILE, stored_features)
                return jsonify({"message": "New features added to the database."}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Allowed formats are: png, jpg, jpeg."}), 400

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
