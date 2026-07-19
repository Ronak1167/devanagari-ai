# /// script
# dependencies = [
#   "flask",
#   "numpy",
#   "pillow",
#   "torch",
#   "torchvision",
#   "scikit-learn",
#   "joblib"
# ]
# ///
import os
import re
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")

project_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Model paths ─────────────────────────────────────────────────────────────
CNN_MODEL_PATH  = os.path.join(project_dir, "models", "devanagari_cnn_model.pth")
CNN_CLASS_PATH  = os.path.join(project_dir, "models", "cnn_class_names.json")

model_payload   = None
model_type      = None   # "cnn" | "rf"

# ─── Character mapping ────────────────────────────────────────────────────────
MAPPING = {
    "character_1_ka":          {"char": "क",  "name": "ka"},
    "character_2_kha":         {"char": "ख",  "name": "kha"},
    "character_3_ga":          {"char": "ग",  "name": "ga"},
    "character_4_gha":         {"char": "घ",  "name": "gha"},
    "character_5_kna":         {"char": "ङ",  "name": "kna"},
    "character_6_cha":         {"char": "च",  "name": "cha"},
    "character_7_chha":        {"char": "छ",  "name": "chha"},
    "character_8_ja":          {"char": "ज",  "name": "ja"},
    "character_9_jha":         {"char": "झ",  "name": "jha"},
    "character_10_yna":        {"char": "ञ",  "name": "yna"},
    "character_11_taamatar":   {"char": "ट",  "name": "ta"},
    "character_12_thaa":       {"char": "ठ",  "name": "tha"},
    "character_13_daa":        {"char": "ड",  "name": "da"},
    "character_14_dhaa":       {"char": "ढ",  "name": "dha"},
    "character_15_adna":       {"char": "ण",  "name": "adna"},
    "character_16_tabala":     {"char": "त",  "name": "ta"},
    "character_17_tha":        {"char": "थ",  "name": "tha"},
    "character_18_da":         {"char": "द",  "name": "da"},
    "character_19_dha":        {"char": "ध",  "name": "dha"},
    "character_20_na":         {"char": "न",  "name": "na"},
    "character_21_pa":         {"char": "प",  "name": "pa"},
    "character_22_pha":        {"char": "फ",  "name": "pha"},
    "character_23_ba":         {"char": "ब",  "name": "ba"},
    "character_24_bha":        {"char": "भ",  "name": "bha"},
    "character_25_ma":         {"char": "म",  "name": "ma"},
    "character_26_yaw":        {"char": "य",  "name": "ya"},
    "character_27_ra":         {"char": "र",  "name": "ra"},
    "character_28_la":         {"char": "ल",  "name": "la"},
    "character_29_waw":        {"char": "व",  "name": "va"},
    "character_30_motosaw":    {"char": "श",  "name": "sha"},
    "character_31_petchiryakha": {"char": "ष", "name": "sha"},
    "character_32_patalosaw":  {"char": "स",  "name": "sa"},
    "character_33_ha":         {"char": "ह",  "name": "ha"},
    "character_34_chhya":      {"char": "क्ष","name": "kshya"},
    "character_35_tra":        {"char": "त्र","name": "tra"},
    "character_36_gya":        {"char": "ज्ञ","name": "gya"},
    "digit_0":                 {"char": "०",  "name": "0"},
    "digit_1":                 {"char": "१",  "name": "1"},
    "digit_2":                 {"char": "२",  "name": "2"},
    "digit_3":                 {"char": "३",  "name": "3"},
    "digit_4":                 {"char": "४",  "name": "4"},
    "digit_5":                 {"char": "५",  "name": "5"},
    "digit_6":                 {"char": "६",  "name": "6"},
    "digit_7":                 {"char": "७",  "name": "7"},
    "digit_8":                 {"char": "८",  "name": "8"},
    "digit_9":                 {"char": "९",  "name": "9"},
}

# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    global model_payload, model_type

    if model_payload is not None:
        return  # Already loaded

    # Try CNN first
    if os.path.exists(CNN_MODEL_PATH) and os.path.exists(CNN_CLASS_PATH):
        try:
            import torch
            from scripts.train_cnn import DevanagariCNN

            with open(CNN_CLASS_PATH) as f:
                class_names = json.load(f)

            checkpoint = torch.load(CNN_MODEL_PATH, map_location='cpu', weights_only=False)
            cnn = DevanagariCNN(num_classes=len(class_names))
            cnn.load_state_dict(checkpoint['model_state_dict'])
            cnn.eval()

            model_payload = {"model": cnn, "classes": class_names}
            model_type = "cnn"
            print(f"[OK] CNN model loaded - Val acc: {checkpoint.get('val_acc', 0):.2f}%")
            return
        except Exception as e:
            print(f"CNN load failed ({e}), trying Random Forest fallback...")

    # Fallback to Random Forest
    if os.path.exists(RF_MODEL_PATH):
        try:
            import joblib
            payload = joblib.load(RF_MODEL_PATH)
            model_payload = payload
            model_type = "rf"
            print("[OK] Random Forest model loaded (fallback).")
            return
        except Exception as e:
            print(f"RF load also failed: {e}")

    print("[ERROR] No model found. Run train_cnn.py first.")


# ─── Image preprocessing (shared) ────────────────────────────────────────────

def otsu_threshold(gray_img):
    pixel_counts, bin_edges = np.histogram(gray_img, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(pixel_counts)
    weight2 = np.cumsum(pixel_counts[::-1])[::-1]
    weight1 = np.where(weight1 == 0, 1, weight1)
    weight2 = np.where(weight2 == 0, 1, weight2)
    mean1 = np.cumsum(pixel_counts * bin_centers) / weight1
    mean2 = (np.cumsum((pixel_counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
    return bin_centers[np.argmax(variance_between)]


def preprocess_canvas(image_data_b64) -> np.ndarray:
    """
    Returns a 32x32 numpy array (float32, range 0–1) suitable for either model.
    White strokes on black background, character tightly cropped + padded, aspect preserved.
    """
    img = Image.open(BytesIO(base64.b64decode(image_data_b64)))

    if img.mode == 'RGBA':
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert('L')
    else:
        img = img.convert('L')

    np_img = np.array(img, dtype=np.float32)

    # Invert if light background (uploaded paper scans)
    border = np.concatenate([np_img[0,:], np_img[-1,:], np_img[:,0], np_img[:,-1]])
    if np.mean(border) > 127:
        np_img = 255.0 - np_img

    # Contrast stretch
    mn, mx = np_img.min(), np_img.max()
    if mx > mn:
        np_img = (np_img - mn) / (mx - mn) * 255.0

    # Otsu threshold
    thresh = max(otsu_threshold(np_img), 30.0)
    binary_np = np.where(np_img > thresh, 255, 0).astype(np.uint8)

    non_zero = np.argwhere(binary_np > 10)

    pil_bin = Image.fromarray(binary_np)
    pil_bin = pil_bin.filter(ImageFilter.GaussianBlur(radius=1.0))

    if non_zero.size > 0:
        min_y, min_x = non_zero.min(axis=0)
        max_y, max_x = non_zero.max(axis=0)
        cropped = pil_bin.crop((min_x, min_y, max_x, max_y))

        w, h = cropped.size
        if w > h:
            pad = (w - h)
            cropped = ImageOps.expand(cropped, border=(0, pad//2, 0, pad - pad//2), fill=0)
        elif h > w:
            pad = (h - w)
            cropped = ImageOps.expand(cropped, border=(pad//2, 0, pad - pad//2, 0), fill=0)

        sq = cropped.size[0]
        cropped = ImageOps.expand(cropped, border=int(sq * 0.10), fill=0)
        padded = cropped
    else:
        padded = pil_bin

    resized = padded.resize((32, 32), Image.BILINEAR)
    
    # Recover faint lines cleanly without causing blobby merges
    np_res = np.array(resized)
    np_res = np.where(np_res > 100, 255, 0).astype(np.uint8)
    resized = Image.fromarray(np_res)
    
    # DEBUG: Save the final processed image to see what the model actually receives
    debug_path = r'C:\Users\Ronak Jain\.gemini\antigravity-ide\brain\75be460a-1002-4fa4-a052-17546b9b77fe\scratch\canvas_debug.png'
    resized.save(debug_path)
    
    return np.array(resized, dtype=np.float32) / 255.0


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    if model_payload is None:
        return jsonify({"error": "Model not found. Run train_cnn.py first."}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided."}), 400

    image_data = re.sub(r'^data:image/.+;base64,', '', data['image'])

    try:
        arr_32x32 = preprocess_canvas(image_data)  # shape (32, 32), range [0,1]

        classes = model_payload['classes']
        model   = model_payload['model']

        if model_type == "cnn":
            import torch
            from torchvision import transforms
            # Normalize to [-1, 1] like the training transform
            tensor = torch.tensor(arr_32x32, dtype=torch.float32)
            tensor = (tensor - 0.5) / 0.5            # [-1, 1]
            tensor = tensor.unsqueeze(0).unsqueeze(0) # (1, 1, 32, 32)

            with torch.no_grad():
                logits = model(tensor)                # (1, num_classes)
                probs  = torch.softmax(logits, dim=1).squeeze().numpy()

        else:  # Random Forest
            features = arr_32x32.flatten().reshape(1, -1)
            probs = model.predict_proba(features)[0]

        top_indices = np.argsort(probs)[::-1][:3]

        predictions = []
        for idx in top_indices:
            class_name = classes[idx]
            mapped = MAPPING.get(class_name, {"char": "?", "name": class_name})
            predictions.append({
                "class_name":  class_name,
                "character":   mapped["char"],
                "phonetic":    mapped["name"],
                "confidence":  float(probs[idx])
            })

        return jsonify({
            "predictions":  predictions,
            "model_type":   model_type,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(host='127.0.0.1', port=5000, debug=False)
