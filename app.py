"""
==============================================================================
FloraVision AI 2.0 - Botanical Intelligence & Flower Classification Server
==============================================================================
Flask backend serving real-time PyTorch MobileNetV2 image classification,
verified botanical taxonomy, responsible medicinal/traditional information,
technical image quality checks, and Grad-CAM neural explainability.

Supported Classes (5):
  - Hibiscus   (Family: Malvaceae)
  - Rose       (Family: Rosaceae)
  - Sunflower  (Family: Asteraceae)
  - Lotus      (Family: Nelumbonaceae)
  - Iris       (Family: Iridaceae)

Conservative Rejection Safeguard:
  - Predictions with maximum softmax probability < 0.55 (55%) are flagged as
    uncertain/out-of-distribution, suppressing taxonomy to avoid misinformation.
==============================================================================
"""

import os
import io
import json
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

# Initialize Flask
app = Flask(__name__)

# Security & Upload Configuration
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "flower_classifier.pt")
METADATA_PATH = os.path.join(BASE_DIR, "metadata", "flower_metadata.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Conservative Rejection / Uncertainty Threshold
# With 5 classes, random baseline is 0.20 (20%).
# Predictions below 0.55 (55%) are flagged as uncertain / unsupported.
CONFIDENCE_THRESHOLD = 0.55

# Global Resources
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

        # Reconstruct MobileNetV2 architecture with matching linear head
        m = models.mobilenet_v2(weights=None)
        in_features = m.last_channel
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        m.load_state_dict(checkpoint["state_dict"])
        m.eval()

        model = m

        # Preprocessing pipeline matching training specification
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


# Attempt loading on module import
load_resources_at_startup()


def allowed_file(filename):
    """
    Validates file extension against allowed formats.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image_quality(pil_img, image_bytes):
    """
    Performs objective technical checks on uploaded image:
    - Dimensions & Aspect Ratio
    - Average Luminance (Exposure: Underexposed vs Overexposed)
    - File size
    """
    w, h = pil_img.size
    filesize_kb = round(len(image_bytes) / 1024, 1)

    gray = pil_img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_lum = stat.mean[0]
    std_lum = stat.stddev[0]

    notes = []
    is_optimal = True

    if w < 160 or h < 160:
        notes.append(f"Low resolution ({w}×{h} px). Fine petal structures may be degraded.")
        is_optimal = False

    if mean_lum < 25:
        notes.append(f"Underexposed / very dark (avg luminance {mean_lum:.1f}/255). Visual detail may be obscured.")
        is_optimal = False
    elif mean_lum > 240:
        notes.append(f"Overexposed / bleached (avg luminance {mean_lum:.1f}/255). Petal colors may be washed out.")
        is_optimal = False

    if not notes:
        notes.append("Optimal exposure and resolution for convolutional feature extraction.")

    return {
        "is_optimal": is_optimal,
        "width": w,
        "height": h,
        "aspect_ratio": f"{w}:{h}",
        "filesize_kb": filesize_kb,
        "luminance_avg": round(mean_lum, 1),
        "luminance_std": round(std_lum, 1),
        "notes": notes
    }


def compute_gradcam(model_ref, input_tensor, original_pil_img, target_class_idx):
    """
    Computes genuine Grad-CAM attention heatmap from MobileNetV2's final conv layer.
    Overlays attention heatmap on the input image as a Base64 data URL.
    """
    try:
        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = model_ref.features[-1]
        h_f = target_layer.register_forward_hook(forward_hook)
        h_b = target_layer.register_full_backward_hook(backward_hook)

        x = input_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model_ref(x)
            model_ref.zero_grad()
            score = logits[0, target_class_idx]
            score.backward()

        h_f.remove()
        h_b.remove()

        if not activations or not gradients:
            return None

        act = activations[0]  # [1, 1280, 7, 7]
        grad = gradients[0]   # [1, 1280, 7, 7]

        # Global average pool the gradients to obtain feature importance weights
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)
        cam = nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max > c_min:
            cam_norm = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Pure NumPy jet colormap (no matplotlib runtime dependency required)
        x_val = 4.0 * cam_norm
        r = np.clip(1.5 - np.abs(x_val - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(x_val - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(x_val - 1.0), 0.0, 1.0)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        # Resize heatmap to match original image size
        heatmap_pil = Image.fromarray(rgb).resize(original_pil_img.size, Image.Resampling.BILINEAR)

        # Blend original with 45% heatmap transparency
        blended = Image.blend(original_pil_img.convert("RGB"), heatmap_pil, alpha=0.45)

        buf = io.BytesIO()
        blended.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    except Exception as e:
        print(f"[!] Grad-CAM computation note: {e}")
        return None


def preprocess_image(image_bytes):
    """
    Validates, decodes, and preprocesses uploaded image bytes into a PyTorch tensor.
    Returns: (input_tensor, pil_img)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise ValueError("Uploaded file is corrupted or not a valid image.")

    # Re-open after verify()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    if image_transforms is None:
        raise RuntimeError("Preprocessing transforms are not initialized.")

    tensor = image_transforms(img)
    tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor, img


@app.route("/", methods=["GET"])
def home():
    """
    Serves the main application interface.
    """
    return render_template("index.html")


@app.route("/results/<path:filename>", methods=["GET"])
def serve_results(filename):
    """
    Serves generated evaluation artifacts (e.g. confusion matrix plot).
    """
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction & Botanical Intelligence API endpoint.
    Accepts multipart/form-data containing 'image'.
    Returns structured JSON with top predictions, botanical intelligence report,
    technical image quality diagnostics, and Grad-CAM neural explainability.
    """
    global model

    # 1. Verify model availability
    if model is None:
        success = load_resources_at_startup()
        if not success or model is None:
            return jsonify({
                "status": "error",
                "error": "The flower classification model is not available. Please verify model weights."
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

    # 4. Read bytes & analyze technical quality
    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({
                "status": "error",
                "error": "Uploaded image file is empty (0 bytes)."
            }), 400

        input_tensor, pil_img = preprocess_image(image_bytes)
        quality_info = analyze_image_quality(pil_img, image_bytes)

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception:
        return jsonify({
            "status": "error",
            "error": "Failed to process the uploaded image. Please ensure it is a valid photo."
        }), 400

    # 5. Run neural network inference
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
                    "probability_pct": f"{prob_val * 100:.1f}%",
                    "class_index": i
                })

            # Sort descending by probability
            prob_list.sort(key=lambda x: x["probability"], reverse=True)

            best_pred = prob_list[0]
            max_prob = best_pred["probability"]
            predicted_class = best_pred["class_code"]
            winning_idx = best_pred["class_index"]

            # 6. Conservative Rejection Safeguard
            if max_prob < CONFIDENCE_THRESHOLD:
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
                    "image_quality": quality_info,
                    "botanical_info": None,
                    "gradcam_heatmap": None,
                    "model_info": {
                        "architecture": "MobileNetV2",
                        "framework": "PyTorch 2.x",
                        "input_resolution": "224×224 RGB",
                        "supported_classes": 5,
                        "rejection_threshold": f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
                    }
                }), 200

            # 7. Accepted Prediction & Botanical Intelligence Report
            botanical_facts = metadata_db.get(predicted_class, {}) or {}

            # Optional Grad-CAM explainability
            gradcam_url = compute_gradcam(model, input_tensor, pil_img, winning_idx)

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

                # Enriched Botanical Intelligence Fields
                "overview": botanical_facts.get("overview", botanical_facts.get("characteristics", "")),
                "key_characteristics": botanical_facts.get("key_characteristics", botanical_facts.get("characteristics", "")),
                "geographic_distribution": botanical_facts.get("geographic_distribution", botanical_facts.get("habitat", "")),
                "ecological_importance": botanical_facts.get("ecological_importance", ""),
                "common_uses": botanical_facts.get("common_uses", {}),
                "traditional_medicinal_info": botanical_facts.get("traditional_medicinal_info", {}),
                "authoritative_sources": botanical_facts.get("authoritative_sources", []),

                "probability": max_prob,
                "confidence_pct": round(max_prob * 100, 2),
                "probability_pct": f"{max_prob * 100:.1f}%",
                "message": f"Successfully identified as {best_pred['flower']}.",
                "top_predictions": prob_list,
                "botanical_info": botanical_facts,
                "image_quality": quality_info,
                "gradcam_heatmap": gradcam_url,
                "model_info": {
                    "architecture": "MobileNetV2",
                    "framework": "PyTorch 2.x",
                    "input_resolution": "224×224 RGB",
                    "supported_classes": 5,
                    "rejection_threshold": f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
                }
            }), 200

    except Exception as e:
        print(f"[!] Inference runtime error: {e}")
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
    port = int(os.environ.get("PORT", 5000))
    print("[*] Starting FloraVision AI 2.0 Server...")
    print(f"[*] Rejection threshold set to: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"[*] Server listening on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
