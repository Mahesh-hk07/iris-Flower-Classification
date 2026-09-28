"""
==============================================================================
Multi-Class Botanical Flower Dataset Downloader
==============================================================================
This utility downloads legitimate botanical photographs of five distinct
flower classes from Wikimedia Commons under Creative Commons / Public Domain
licenses:
  1. Hibiscus   (Family: Malvaceae)
  2. Rose       (Family: Rosaceae)
  3. Sunflower  (Family: Asteraceae)
  4. Lotus      (Family: Nelumbonaceae)
  5. Iris       (Family: Iridaceae)

Splits images into:
  - data/train/<class>/       (70%)
  - data/validation/<class>/  (15%)
  - data/test/<class>/        (15%)

And creates data/DATASET_INFO.md with complete licensing & source attribution.
==============================================================================
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from PIL import Image
import io
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

FLOWER_CATEGORIES = {
    "hibiscus": "Category:Hibiscus_rosa-sinensis",
    "rose": "Category:Rosa_canina",
    "sunflower": "Category:Helianthus_annuus",
    "lotus": "Category:Nelumbo_nucifera",
    "iris": "Category:Iris_setosa"
}

IMAGES_PER_CLASS = 14
TRAIN_RATIO = 0.70  # ~10 train
VAL_RATIO = 0.15    # ~2 val, ~2 test

USER_AGENT = "MultiClassFlowerClassifier/2.0 (educational research bot; mayab@example.com)"


def fetch_category_images(category_name, limit=35):
    """
    Queries Wikimedia Commons API for image files within a category.
    """
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_name,
        "cmtype": "file",
        "cmlimit": str(limit),
        "format": "json"
    }
    url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        members = data.get("query", {}).get("categorymembers", [])
        return [
            m["title"] for m in members
            if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))
            and not any(bad in m["title"].lower() for bad in ["diagram", "map", "chart", "drawing", "illustration", "herbarium", "sheet", "painting"])
        ]


def fetch_image_info(title):
    """
    Fetches direct thumbnail URL, author, and license for a Wikimedia Commons file.
    Uses iiurlwidth=500 for fast, lightweight botanical photos.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": "500",
        "format": "json"
    }
    url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            infos = page.get("imageinfo", [])
            if infos:
                info = infos[0]
                download_url = info.get("thumburl") or info.get("url")
                metadata = info.get("extmetadata", {})
                license_name = metadata.get("LicenseShortName", {}).get("value", "Public Domain / CC")
                artist = metadata.get("Artist", {}).get("value", "Wikimedia Contributor")
                if "<" in artist:
                    import re
                    artist = re.sub(r'<[^>]+>', '', artist).strip()
                return {
                    "title": title,
                    "url": download_url,
                    "license": license_name,
                    "artist": artist[:50]
                }
    return None


def download_and_prepare_dataset(clean_existing=True):
    print("=" * 68)
    print("  DOWNLOADING MULTI-CLASS BOTANICAL FLOWER PHOTOGRAPHS")
    print("  Source: Wikimedia Commons (Creative Commons / Public Domain)")
    print("  Target Classes: Hibiscus, Rose, Sunflower, Lotus, Iris")
    print("=" * 68)

    if clean_existing and os.path.exists(DATA_DIR):
        print("[*] Refreshing data/ directory...")
        shutil.rmtree(DATA_DIR, ignore_errors=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = []

    train_limit = int(IMAGES_PER_CLASS * TRAIN_RATIO)  # 9-10
    val_limit = train_limit + int(IMAGES_PER_CLASS * VAL_RATIO)  # 11-12

    for flower_name, category in FLOWER_CATEGORIES.items():
        print(f"\n[*] Querying category for {flower_name.upper()} ({category})...")
        train_dir = os.path.join(DATA_DIR, "train", flower_name)
        val_dir = os.path.join(DATA_DIR, "validation", flower_name)
        test_dir = os.path.join(DATA_DIR, "test", flower_name)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        try:
            time.sleep(0.4)  # Politeness delay to prevent rate limits
            titles = fetch_category_images(category, limit=35)
            print(f"    Found {len(titles)} candidate botanical photos.")
        except Exception as e:
            print(f"    [!] Error querying category {category}: {e}")
            continue

        downloaded_count = 0
        for title in titles:
            if downloaded_count >= IMAGES_PER_CLASS:
                break

            try:
                time.sleep(0.2)
                info = fetch_image_info(title)
                if not info or not info.get("url"):
                    continue

                img_url = info["url"]
                img_req = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(img_req, timeout=12) as img_resp:
                    img_bytes = img_resp.read()

                # Open with Pillow, convert to RGB, and standardize dimensions
                image = Image.open(io.BytesIO(img_bytes))
                image = image.convert("RGB")
                image.thumbnail((500, 500), Image.Resampling.LANCZOS)

                # Determine split: train (0 to train_limit), val (train_limit to val_limit), test (val_limit to end)
                if downloaded_count < train_limit:
                    target_folder = train_dir
                    split_name = "train"
                elif downloaded_count < val_limit:
                    target_folder = val_dir
                    split_name = "validation"
                else:
                    target_folder = test_dir
                    split_name = "test"

                filename = f"{flower_name}_{downloaded_count + 1:02d}.jpg"
                save_path = os.path.join(target_folder, filename)
                image.save(save_path, "JPEG", quality=88)

                downloaded_count += 1
                manifest.append({
                    "class": flower_name,
                    "split": split_name,
                    "file": filename,
                    "title": info["title"],
                    "url": info["url"],
                    "license": info["license"],
                    "artist": info["artist"]
                })
                print(f"    [{downloaded_count}/{IMAGES_PER_CLASS}] Saved {flower_name}/{filename} ({split_name}) - {info['license']}")

            except Exception:
                continue

        print(f"[*] Completed {flower_name.upper()}: {downloaded_count} photos stored.")

    # Write data/DATASET_INFO.md
    manifest_path = os.path.join(DATA_DIR, "DATASET_INFO.md")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# Botanical Multi-Class Flower Image Dataset\n\n")
        f.write("**Source**: Wikimedia Commons (Biodiversity Categories)\n\n")
        f.write("**Supported Classes & Botanical Families**:\n")
        f.write("- `hibiscus`: *Hibiscus rosa-sinensis* (Family: **Malvaceae**)\n")
        f.write("- `rose`: *Rosa canina* (Family: **Rosaceae**)\n")
        f.write("- `sunflower`: *Helianthus annuus* (Family: **Asteraceae**)\n")
        f.write("- `lotus`: *Nelumbo nucifera* (Family: **Nelumbonaceae**)\n")
        f.write("- `iris`: *Iris setosa* (Family: **Iridaceae**)\n\n")
        f.write(f"**Total Verified Images**: {len(manifest)}\n\n")
        f.write("| Class | Split | Filename | Wikimedia Title | License | Author |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for item in manifest:
            f.write(f"| {item['class'].capitalize()} | {item['split']} | `{item['file']}` | {item['title']} | {item['license']} | {item['artist']} |\n")

    print("\n" + "=" * 68)
    print(f"[*] Dataset catalog written to: {manifest_path}")
    print(f"[*] Total images collected across 5 classes: {len(manifest)}")
    print("=" * 68)
    return len(manifest)


if __name__ == "__main__":
    count = download_and_prepare_dataset(clean_existing=True)
    if count == 0:
        print("[!] Error: No images downloaded. Please check internet connection.")
        sys.exit(1)
    else:
        print(f"[OK] Successfully collected and split {count} real flower photographs!")
