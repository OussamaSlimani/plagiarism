from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import numpy as np
from scipy.spatial.distance import cosine
import os

app = Flask(__name__)

# Load the ResNet-18 model pre-trained on ImageNet with updated weights
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # Set the model to evaluation mode

# Define a transformation to preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path)

    # If the image has an alpha channel (RGBA), convert it to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode == 'L':  # If the image is grayscale, convert it to RGB
        image = image.convert('RGB')

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient computation for inference
        features = model(image)
    
    # Use the output of the penultimate layer (before final classification layer)
    return features.squeeze().numpy()  # Convert tensor to numpy array

# Function to calculate cosine similarity between two feature vectors
def cosine_similarity(feature1, feature2):
    return 1 - cosine(feature1, feature2)

# Function to check and store new features in features.json
def check_and_store_features(image_path, features_file='features.json'):
    # Extract features from the new image
    new_features = extract_features(image_path)
    
    # Load existing features from the JSON file
    stored_features = []

    try:
        with open(features_file, 'r') as f:
            stored_features = json.load(f)
    except FileNotFoundError:
        # If the file does not exist, initialize it with an empty list
        stored_features = []
    except json.JSONDecodeError:
        # If there is an error in decoding (file is empty or corrupt), initialize with an empty list
        print(f"Warning: {features_file} is empty or corrupt. Initializing with an empty list.")
        stored_features = []

    # Compare the new features with stored ones
    for stored_feature in stored_features:
        similarity = cosine_similarity(new_features, np.array(stored_feature))
        if similarity >= 0.9:
            print("Plagiarism detected.")
            return True  # Return True if plagiarism is detected
    
    # If no match found, store the new features in the JSON file
    stored_features.append(new_features.tolist())  # Store as a list of lists
    with open(features_file, 'w') as f:
        json.dump(stored_features, f)
    print("New features added.")
    return False  # No plagiarism

# API to upload image and check for plagiarism
@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image temporarily
    image_path = os.path.join('uploads', image_file.filename)
    os.makedirs('uploads', exist_ok=True)
    image_file.save(image_path)

    # Check for plagiarism
    plagiarism_detected = check_and_store_features(image_path)

    if plagiarism_detected:
        return jsonify({"message": "Plagiarism detected"}), 200
    else:
        return jsonify({"message": "No plagiarism detected, features stored"}), 200

# API to retrieve stored features (optional, for testing)
@app.route('/stored_features', methods=['GET'])
def stored_features():
    try:
        with open('features.json', 'r') as f:
            features = json.load(f)
        return jsonify(features), 200
    except FileNotFoundError:
        return jsonify({"message": "No stored features found."}), 404

if __name__ == '__main__':
    app.run(debug=True)
