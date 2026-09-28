# FloraVision AI: Multi-Class Botanical Flower Image Classifier

> A genuine computer-vision web application built with **PyTorch**, **MobileNetV2 Transfer Learning**, and **Flask**. Supports multi-class flower classification (**Hibiscus**, **Rose**, **Sunflower**, **Lotus**, **Iris**) from uploaded photographs, with verified botanical taxonomy and conservative out-of-distribution rejection.

---

## 1. Project Overview & Motivation
A critical flaw in naive machine-learning vision apps is the "closed-world" assumption: if a model is trained only on *Iris* images, it will blindly force an unrelated flower photograph (such as a *Hibiscus* or *Rose*) to be classified as an *Iris* (e.g. *Iris versicolor* with ~36% probability). 

This application implements a complete, modern computer vision pipeline:
1. **Multi-Class Botanical Coverage**: Trained on genuine photographs of 5 distinct flower classes spanning 5 separate botanical families (*Malvaceae*, *Rosaceae*, *Asteraceae*, *Nelumbonaceae*, *Iridaceae*).
2. **Transfer Learning with MobileNetV2**: Employs deep inverted residual bottleneck layers pretrained on ImageNet-1K with botanical domain adaptation on high-level feature blocks.
3. **Verified Botanical Metadata**: Flower families, taxonomic orders, and scientific names are retrieved strictly from a verified local database (`metadata/flower_metadata.json`), ensuring zero hallucination.
4. **Conservative Rejection Mechanism**: A 40.0% confidence cutoff rejects low-probability or non-flower images as `"uncertain / unsupported"`, suppressing botanical metadata rather than forcing an inaccurate prediction.
5. **No Tabular or Numerical Measurements**: The application accepts solely real photographic inputs (`.jpg`, `.jpeg`, `.png`, `.webp`) and processes raw RGB pixels through deep convolutional layers.

---

## 2. Supported Classes & Botanical Taxonomy

| Class Key | Display Name | Scientific Name | Botanical Family | Order |
| :--- | :--- | :--- | :--- | :--- |
| `hibiscus` | **Hibiscus** | *Hibiscus rosa-sinensis* | **Malvaceae** | Malvales |
| `rose` | **Rose** | *Rosa chinensis / Rosa spp.* | **Rosaceae** | Rosales |
| `sunflower` | **Sunflower** | *Helianthus annuus* | **Asteraceae** | Asterales |
| `lotus` | **Lotus** | *Nelumbo nucifera* | **Nelumbonaceae** | Proteales |
| `iris` | **Iris** | *Iris setosa / Iris spp.* | **Iridaceae** | Asparagales |

---

## 3. Computer Vision Architecture: MobileNetV2 Transfer Learning

```
Input Image (JPG/PNG/WEBP)
       │
       ▼
[ Center Crop & Resize to 224x224 RGB ]
       │
       ▼
[ ImageNet Normalization: μ=(0.485, 0.456, 0.406), σ=(0.229, 0.224, 0.225) ]
       │
       ▼
[ MobileNetV2 Pretrained Convolutional Backbone ]
   ├── features[0..15] : Frozen low-level feature extractors (edges, textures)
   └── features[16..18] : Unfrozen top feature block (Botanical fine-tuning, lr=1e-4)
       │
       ▼
[ Global Adaptive Average Pooling (1280-dim representation vector) ]
       │
       ▼
[ Custom Classification Head (lr=1e-3) ]
   ├── Dropout(p=0.2)
   └── Linear(in_features=1280, out_features=5 classes)
       │
       ▼
[ Logits ] ──▶ [ Softmax ] ──▶ Probabilities (Hibiscus, Rose, Sunflower, Lotus, Iris)
```

### Why PyTorch MobileNetV2?
- **Efficiency**: Depthwise separable convolutions provide high inference speed (~40 ms per image on standard laptop CPU) with a compact checkpoint footprint (~8.7 MB).
- **Fine-Tuning**: Unfreezing top convolutional blocks enables the network to learn nuanced botanical traits (e.g. hibiscus stamens, rose petal whorls, sunflower disc florets).

---

## 4. Conservative Rejection Mechanism

For a 5-class classifier, uniform random guessing yields $1/5 = 20\%$. An image of an unsupported flower species or non-flower object produces diffuse, ambiguous probabilities.

The backend enforces a configurable rejection threshold (`CONFIDENCE_THRESHOLD = 0.40`):
$$\text{If } \max_{i} P(\text{class}_i) < 0.40 \implies \text{Status: "uncertain" (Rejected)}$$

When an image is rejected:
- Status is returned as `"uncertain"`.
- `predicted_class` and `botanical_family` are returned as `null`.
- The frontend suppresses the botanical taxonomy card to avoid misleading the user.
- Candidate probabilities are shown transparently for diagnostic insight.

---

## 5. Measured Model Evaluation & Results

Evaluated against the held-out test dataset in `data/test/`:
- **Training Accuracy**: 97.87% (Loss: 0.2304)
- **Validation Accuracy**: 70.00%
- **Held-Out Test Accuracy**: 56.25% (9/16 correct on independent Wikimedia test images)
- **Rose Test Performance**: Precision 0.60, Recall 0.75, F1-Score 0.67
- **Confusion Matrix**: Generated and saved to [`results/confusion_matrix.png`](file:///c:/Users/mayab/OneDrive/Desktop/iris%20Flower%20Classification/results/confusion_matrix.png)

### Live Sample Verification:
- **Hibiscus** (`sample_hibiscus.jpg`) $\rightarrow$ Predicted **Hibiscus** (*Malvaceae*), **97.4%** confidence (Match).
- **Rose** (`sample_rose.jpg`) $\rightarrow$ Predicted **Rose** (*Rosaceae*), **93.9%** confidence (Match).
- **Sunflower** (`sample_sunflower.jpg`) $\rightarrow$ Predicted **Sunflower** (*Asteraceae*), **89.9%** confidence (Match).
- **Lotus** (`sample_lotus.jpg`) $\rightarrow$ Predicted **Lotus** (*Nelumbonaceae*), **62.7%** confidence (Match).
- **Iris** (`sample_iris.jpg`) $\rightarrow$ Predicted **Iris** (*Iridaceae*), **99.9%** confidence (Match).
- **Uncertain / Non-Flower** (`grey_noise_test.jpg`) $\rightarrow$ Top prob **38.2%** (< 40.0%) $\rightarrow$ Correctly rejected as **Uncertain**.

---

## 6. Project Structure

```
Iris Flower Classification/
│
├── app.py                     # Flask web server & REST prediction API
├── train_model.py             # MobileNetV2 transfer learning training script
├── train.py                   # Model training entrypoint (mirrors train_model.py)
├── evaluate_model.py          # Held-out test evaluation entrypoint
├── evaluate.py                # Standalone test-set evaluation & confusion matrix
├── download_dataset.py        # Automated Wikimedia Commons photo downloader
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Git ignore configuration
│
├── data/
│   ├── DATASET_INFO.md        # Image catalog, licenses, and sources
│   ├── train/                 # 47 training images across 5 classes
│   ├── validation/            # 10 validation images
│   └── test/                  # 16 held-out test images
│
├── metadata/
│   └── flower_metadata.json   # Verified botanical taxonomy database
│
├── models/
│   └── flower_classifier.pt   # Trained PyTorch MobileNetV2 checkpoint (8.74 MB)
│
├── results/
│   └── confusion_matrix.png   # Measured test confusion matrix plot
│
├── templates/
│   └── index.html             # Modern, attractive glassmorphic user interface
│
└── static/
    ├── style.css              # Vanilla CSS3 styling (dark glassmorphic theme)
    ├── script.js              # Vanilla JavaScript (drag-and-drop, preview, fetch)
    └── samples/               # Quick-test botanical photographs
```

---

## 7. Installation & Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Download / Refresh Dataset (Optional)
```powershell
python download_dataset.py
```

### 3. Train the Transfer Learning CNN
```powershell
python train_model.py
```

### 4. Evaluate on Held-Out Test Set
```powershell
python evaluate_model.py
```

### 5. Launch the Web Application
```powershell
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.
