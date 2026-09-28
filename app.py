"""
==============================================================================
Multi-Class Flower Image Classification - Flask Web Application
==============================================================================
This module serves the web application for multi-class flower photo classification.
It loads the PyTorch MobileNetV2 model checkpoint (models/flower_classifier.pt)
and verified botanical metadata (metadata/flower_metadata.json) once at startup.

Supported Classes:
  - Hibiscus   (Family: Malvaceae)
  - Rose       (Family: Rosaceae)
  - Sunflower  (Family: Asteraceae)
  - Lotus      (Family: Nelumbonaceae)
  - Iris       (Family: Iridaceae)

Conservative Rejection Mechanism:
  - If the maximum model probability is below CONFIDENCE_THRESHOLD (0.40),
    the model rejects the prediction as 'uncertain' rather than forcing
    an incorrect classification.
==============================================================================
"""

import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Initialize Flask
app = Flask(__name__)

# Security & Upload Configuration
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "flower_classifier.pt")
METADATA_PATH = os.path.join(BASE_DIR, "metadata", "flower_metadata.json")

# Configurable Rejection / Uncertainty Threshold
# With 5 classes, random baseline is 0.20 (20%).
# Predictions below 0.55 (55%) are flagged as uncertain / unsupported.
CONFIDENCE_THRESHOLD = 0.55

# Global State
model = None
class_names = []
metadata_db = {}
image_transforms = None


def load_resources_at_startup():
    """
    Loads model checkpoint and botanical metadata at application launch.
    """
    global model, class_names, metadata_db, image_transforms

    # 1. Load botanical metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_db = json.load(f)
        print(f"[*] Loaded verified botanical metadata for {len(metadata_db)} flower classes.")
    else:
        print(f"[!] Warning: Metadata file not found at {METADATA_PATH}.")
        metadata_db = {}

    # 2. Load trained model
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Warning: Model file not found at {MODEL_PATH}.")
        print("[!] Please run 'python train.py' to train and save the model.")
        return False

    try:
        print(f"[*] Loading trained MobileNetV2 checkpoint from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

        num_classes = checkpoint.get("num_classes", 5)
        class_names = checkpoint.get("class_names", ["hibiscus", "iris", "lotus", "rose", "sunflower"])
        input_size = checkpoint.get("input_size", 224)
        norm_mean = checkpoint.get("norm_mean", [0.485, 0.456, 0.406])
        norm_std = checkpoint.get("norm_std", [0.229, 0.224, 0.225])

        # Reconstruct MobileNetV2 architecture with matching head
        m = models.mobilenet_v2(weights=None)
        in_features = m.last_channel
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        m.load_state_dict(checkpoint["state_dict"])
        m.eval()

        model = m

        # Preprocessing pipeline
        image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

        print(f"[*] Model loaded successfully! Supported classes: {class_names}")
        print(f"[*] Validation Accuracy recorded at training: {checkpoint.get('validation_accuracy', 0.0) * 100:.2f}%")
        return True

    except Exception as e:
        print(f"[!] Error loading model checkpoint: {e}")
        return False


# Attempt loading on module start
load_resources_at_startup()


def allowed_file(filename):
    """
    Validates file extension.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    """
    Validates, decodes, and preprocesses uploaded image bytes.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise ValueError("Uploaded file is corrupted or not a valid image.")

    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    if image_transforms is None:
        raise RuntimeError("Preprocessing transforms are not initialized.")

    tensor = image_transforms(img)
    tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor


@app.route("/", methods=["GET"])
def home():
    """
    Serves the main application interface.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction API endpoint.
    Accepts multipart/form-data containing 'image'.
    Returns structured JSON with top predictions, botanical family, and uncertainty status.
    """
    global model

    # 1. Verify model is loaded
    if model is None:
        success = load_resources_at_startup()
        if not success or model is None:
            return jsonify({
                "status": "error",
                "error": "The flower classification model is not available. Please run 'python train.py' first."
            }), 500

    # 2. Verify file presence
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "error": "No file uploaded. Please select an image file under the 'image' field."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "error": "No file was selected. Please choose an image to upload."
        }), 400

    # 3. Validate file extension
    if not allowed_file(file.filename):
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({
            "status": "error",
            "error": f"Unsupported file format. Please upload an image in one of: {allowed_list}."
        }), 400

    # 4. Preprocess image
    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({
                "status": "error",
                "error": "Uploaded image file is empty (0 bytes)."
            }), 400

        input_tensor = preprocess_image(image_bytes)

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception:
        return jsonify({
            "status": "error",
            "error": "Failed to process the uploaded image. Please ensure it is a valid photo."
        }), 400

    # 5. Run inference
    try:
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

            # Build sorted list of all predictions
            prob_list = []
            for i, c_name in enumerate(class_names):
                display = metadata_db.get(c_name, {}).get("display_name", c_name.capitalize())
                family = metadata_db.get(c_name, {}).get("family", "Unknown")
                prob_val = float(probabilities[i].item())
                prob_list.append({
                    "class": display,
                    "class_code": c_name,
                    "flower": display,
                    "common_name": display,
                    "display_name": display,
                    "family": family,
                    "probability": round(prob_val, 4),
                    "probability_pct": f"{prob_val * 100:.1f}%"
                })

            # Sort descending by probability
            prob_list.sort(key=lambda x: x["probability"], reverse=True)

            best_pred = prob_list[0]
            max_prob = best_pred["probability"]
            predicted_class = best_pred["class_code"]

            # 6. Conservative Rejection / Uncertainty Mechanism
            if max_prob < CONFIDENCE_THRESHOLD:
                # Reject as uncertain / unsupported
                return jsonify({
                    "status": "uncertain",
                    "flower": None,
                    "common_name": None,
                    "predicted_class": None,
                    "family": None,
                    "botanical_family": None,
                    "scientific_name": None,
                    "taxonomic_order": None,
                    "characteristics": None,
                    "habitat": None,
                    "probability": max_prob,
                    "confidence_pct": round(max_prob * 100, 2),
                    "probability_pct": f"{max_prob * 100:.1f}%",
                    "message": (
                        f"The model could not confidently identify this image. "
                        f"Highest model probability ({max_prob * 100:.1f}%) is below the "
                        f"conservative threshold ({CONFIDENCE_THRESHOLD * 100:.0f}%). "
                        "The subject may be an unsupported flower species or non-flower content."
                    ),
                    "top_predictions": prob_list,
                    "botanical_info": None
                }), 200

            # 7. Accepted Prediction
            botanical_facts = metadata_db.get(predicted_class, {}) or {}

            return jsonify({
                "status": "success",
                "flower": best_pred["flower"],
                "common_name": best_pred["flower"],
                "predicted_class": predicted_class,
                "family": best_pred["family"],
                "botanical_family": best_pred["family"],
                "scientific_name": botanical_facts.get("scientific_name", ""),
                "common_names": botanical_facts.get("common_names", ""),
                "full_taxonomy": botanical_facts.get("full_taxonomy", ""),
                "taxonomic_order": botanical_facts.get("order", ""),
                "characteristics": botanical_facts.get("characteristics", ""),
                "habitat": botanical_facts.get("habitat", ""),
                "phenology": botanical_facts.get("phenology", ""),
                "cultural_medicinal": botanical_facts.get("cultural_medicinal", ""),
                "diagnostic_tips": botanical_facts.get("diagnostic_tips", ""),
                "comprehensive_summary": botanical_facts.get("comprehensive_summary", []),
                "probability": max_prob,
                "confidence_pct": round(max_prob * 100, 2),
                "probability_pct": f"{max_prob * 100:.1f}%",
                "message": f"Successfully classified as {best_pred['flower']}.",
                "top_predictions": prob_list,
                "botanical_info": botanical_facts
            }), 200

    except Exception:
        return jsonify({
            "status": "error",
            "error": "An internal error occurred during neural network inference."
        }), 500


# Global HTTP Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "error": "The uploaded file exceeds the 10 MB maximum file size limit."
    }), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/predict"):
        return jsonify({"status": "error", "error": "Endpoint not found."}), 404
    return render_template("index.html"), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "error": f"HTTP method {request.method} is not permitted for this route."
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "error": "An internal server error occurred."
    }), 500


if __name__ == "__main__":
    print("[*] Starting Multi-Class Flower Image Classification Server...")
    print(f"[*] Rejection threshold set to: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print("[*] Open your browser at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
