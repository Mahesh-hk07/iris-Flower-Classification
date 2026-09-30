"""
==============================================================================
FloraVision AI 2.0 - Comprehensive Manual & Regression Verification Suite
==============================================================================
"""

import os
import sys
import json
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, CONFIDENCE_THRESHOLD

def verify_all():
    print("=" * 70)
    print("      FLORAVISION AI 2.0 — COMPREHENSIVE VERIFICATION PASS")
    print("=" * 70)

    client = app.test_client()

    # 1. Homepage & Sections
    r = client.get("/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    html = r.data.decode("utf-8")
    for section_id in ["scanner", "explorer", "pipeline", "model-insights", "limitations", "about"]:
        assert f'id="{section_id}"' in html, f"Missing section id: {section_id}"
    assert "FloraVision" in html
    assert "Confidence Below Configured Threshold" in html
    assert "Technical Check" in html or "Technical Image Quality" in html
    print("[PASS] 1. Homepage & All Core Navigation Anchors")

    # 2. Valid Image Classification (Rose)
    with open("static/samples/sample_rose.jpg", "rb") as f:
        r = client.post("/predict", data={"image": (f, "sample_rose.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200
    data = json.loads(r.data.decode("utf-8"))
    assert data["status"] == "success"
    assert data["flower"] == "Rose"
    assert data["confidence_pct"] > 55.0
    assert data["botanical_family"] == "Rosaceae (Rose Family)"
    assert len(data["top_predictions"]) == 5
    print(f"[PASS] 2. Valid Image Prediction: Rose -> {data['confidence_pct']}% confidence")

    # 3. Grad-CAM Neural Attention Heatmap
    assert data.get("gradcam_heatmap") is not None
    assert data["gradcam_heatmap"].startswith("data:image/jpeg;base64,")
    print(f"[PASS] 3. Grad-CAM Heatmap Generated and Returned as Base64 Payload ({len(data['gradcam_heatmap'])} chars)")

    # 4. Botanical Intelligence Dossier
    assert "overview" in data and len(data["overview"]) > 20
    assert "key_characteristics" in data and len(data["key_characteristics"]) > 20
    assert "geographic_distribution" in data and len(data["geographic_distribution"]) > 20
    assert "ecological_importance" in data and len(data["ecological_importance"]) > 20
    assert "common_uses" in data and len(data["common_uses"]) >= 4
    assert len(data["authoritative_sources"]) >= 3
    print(f"[PASS] 4. Botanical Dossier Complete (Uses: {len(data['common_uses'])}, Sources: {len(data['authoritative_sources'])})")

    # 5. Medicinal Context & Disclaimer
    med = data["traditional_medicinal_info"]
    assert "plant_parts_used" in med
    assert "documented_traditional_use" in med
    assert "scientific_research_context" in med
    assert "evidence_level" in med
    assert "safety_and_disclaimer" in med
    assert "educational" in med["safety_and_disclaimer"].lower()
    print("[PASS] 5. Medicinal Context Responsibly Sourced & Guarded by Disclaimer")

    # 6. Technical Image Quality Diagnostics
    iq = data["image_quality"]
    assert "width" in iq and "height" in iq and "luminance_avg" in iq
    assert iq["width"] == 600 and iq["height"] == 399
    assert iq["is_optimal"] is True
    print(f"[PASS] 6. Technical Image Quality Diagnostics ({iq['width']}x{iq['height']}px, Exposure: {iq['luminance_avg']}/255)")

    # 7. Invalid File Format Rejection
    r = client.post("/predict", data={"image": (io.BytesIO(b"fake txt"), "notes.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400
    err_json = json.loads(r.data.decode("utf-8"))
    assert "Unsupported file format" in err_json["error"]
    print("[PASS] 7. Invalid File Rejection (HTTP 400)")

    # 8. Empty File Rejection
    r = client.post("/predict", data={"image": (io.BytesIO(b""), "empty.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 400
    print("[PASS] 8. Empty File Rejection (HTTP 400)")

    # 9. Low-Confidence Rejection (< 55% threshold)
    with open("static/samples/grey_noise_test.jpg", "rb") as f:
        r = client.post("/predict", data={"image": (f, "grey_noise_test.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200
    d_unc = json.loads(r.data.decode("utf-8"))
    assert d_unc["status"] == "uncertain"
    assert d_unc["flower"] is None
    assert d_unc["botanical_family"] is None
    assert d_unc["confidence_pct"] < 55.0
    assert "below the configured threshold" in d_unc["message"]
    print(f"[PASS] 9. Low-Confidence Rejection & Taxonomy Suppression ({d_unc['confidence_pct']}%)")

    # 10. Confusion Matrix Serving
    r = client.get("/results/confusion_matrix.png")
    assert r.status_code == 200
    assert "image" in r.content_type
    r.close()
    print("[PASS] 10. Confusion Matrix Artifact Accessible (/results/confusion_matrix.png)")

    # 11. All 5 Species Inference Check
    samples = [
        ("sample_hibiscus.jpg", "Hibiscus"),
        ("sample_rose.jpg", "Rose"),
        ("sample_sunflower.jpg", "Sunflower"),
        ("sample_lotus.jpg", "Lotus"),
        ("sample_iris.jpg", "Iris")
    ]
    for s_file, expected_name in samples:
        with open(os.path.join("static", "samples", s_file), "rb") as f:
            r = client.post("/predict", data={"image": (f, s_file)}, content_type="multipart/form-data")
        assert r.status_code == 200
        res = json.loads(r.data.decode("utf-8"))
        assert res["status"] == "success"
        assert res["flower"] == expected_name
        assert res["confidence_pct"] > 55.0
        print(f"       -> {expected_name:10} identified with {res['confidence_pct']}% confidence")
    print("[PASS] 11. All 5 Supported Classes Correctly Inferred with >55% Confidence")

    print("\n" + "=" * 70)
    print("      ALL 11 VERIFICATION STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    verify_all()
