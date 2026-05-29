"""
Flask Web UI — Yog'och Nuqson Klassifikatori
TensorFlow/Keras modeli bilan ishlaydigan lokal deployment
"""

from sklearn.preprocessing import _label
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os
import base64
import matplotlib.cm as cm
from pathlib import Path

app = Flask(__name__)
application = app  # AWS Elastic Beanstalk uchun kerak
CORS(app)  # Barcha originlardan API ga ruxsat

BASE_DIR    = Path(r"c:\Users\yoqub\OneDrive\Desktop\AI 2")
MODEL_PATHS = {
    "EfficientNet":       BASE_DIR / "best_finetuned_fixed.keras",
}

# Yangi o'qitilgan 6 ta klass ro'yxati
CLASS_NAMES = [
    "Crack", 
    "Dead_Knot", 
    "Live_Knot", 
    "knot_with_crack", 
    "resin",
    "unknown"
]

print(f"Klasslar ({len(CLASS_NAMES)} ta): {CLASS_NAMES}")

# ── Modelni yuklash ────────────────────────────────────────────
loaded_models = {}
for name, path in MODEL_PATHS.items():
    if path.exists():
        try:
            loaded_models[name] = tf.keras.models.load_model(str(path))
            print(f"✅ Yuklandi: {name}")
        except Exception as e:
            print(f"❌ Yuklanmadi {name}: {e}")
    else:
        print(f"⚠️ {name} topilmadi: {path}")

ACTIVE_MODEL_NAME = list(loaded_models.keys())[0] if loaded_models else None
print(f"Faol model: {ACTIVE_MODEL_NAME}")


# ── Rasm tayyorlash ────────────────────────────────────────────
def preprocess_image(img_bytes, model_name):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    
    print(f"--- PREPROCESS DEBUG ---")
    print(f"Model Name: {model_name}")
    print(f"Raw Image: Shape={arr.shape}, Min={np.min(arr)}, Max={np.max(arr)}, Mean={np.mean(arr)}")
    
    arr = np.expand_dims(arr, axis=0)
    
    # Normalizatsiya: Faqat Custom CNN uchun 1/255 qilish kerak. 
    # EfficientNet (yangisi) o'zi 0-255 oraliqdagi piksellarni qabul qilib, o'zi ishlaydi.
    if "CNN" in model_name:
        arr = arr / 255.0
        print(f"Rescaled Image for CNN: Min={np.min(arr)}, Max={np.max(arr)}")
        
    return arr

# ── Grad-CAM funksiyalari ──────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, pred_index=None):
    try:
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        if not base_model:
            base_model = model
        
        last_conv_layer_name = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
                
        if not last_conv_layer_name: 
            print("GradCAM: Conv2D qatlami topilmadi!")
            return None
        
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [base_model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None: pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print(f"GradCAM xatosi: {e}")
        return None

def overlay_heatmap(heatmap, original_img_bytes):
    try:
        img = Image.open(io.BytesIO(original_img_bytes)).convert("RGB")
        heatmap = np.uint8(255 * heatmap)
        try:
            jet = cm.get_cmap("jet")
        except AttributeError:
            import matplotlib
            jet = matplotlib.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(img.size)
        superimposed_img = Image.blend(img, jet_heatmap, alpha=0.4)
        buf = io.BytesIO()
        superimposed_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Overlay xatosi: {e}")
        return None

# ── HTML Template ──────────────────────────────────────────────


# ── Routes ─────────────────────────────────────────────────────
from flask import render_template

@app.route("/")
def index():
    return render_template(
        'index.html',
        models=list(loaded_models.keys()) or ["Model topilmadi"],
        n_classes=len(CLASS_NAMES),
        tf_version=tf.__version__
    )

@app.route("/info")
def info():
    return jsonify({
        "models_loaded": len(loaded_models),
        "classes": CLASS_NAMES,
        "n_classes": len(CLASS_NAMES),
        "gpu": len(tf.config.list_physical_devices('GPU')) > 0
    })

@app.route("/predict", methods=["POST"])
def predict_route():
    if not loaded_models:
        return jsonify({"error": "Hech qanday model yuklanmagan! custom_cnn.h5 faylini tekshiring."})

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "Rasm yuklanmadi"})

    model_name = request.form.get("model", list(loaded_models.keys())[0])
    model = loaded_models.get(model_name, list(loaded_models.values())[0])

    try:
        img_bytes = file.read()
        arr = preprocess_image(img_bytes, model_name)

        preds = model.predict(arr, verbose=0)[0]
        print(f"=== Model: {model_name} ===")
        print(f"Preds: {preds}")
        idx   = int(np.argmax(preds))

        # Klass nomlarini model output size ga moslash
        n_out = len(preds)
        names = CLASS_NAMES[:n_out] if len(CLASS_NAMES) >= n_out else \
                CLASS_NAMES + [f"Klass_{i}" for i in range(len(CLASS_NAMES), n_out)]

        # O'zbekchaga o'girish
        UZBEK_NAMES = {
                "Crack": "Yoriq",
                "Dead_Knot": "Quruq ko'z",
                "Live_Knot": "Sog'lom ko'z",
                "knot_with_crack": "Yoriqli ko'z",
                "resin": "Smola",
                "unknown": "Nuqsonsiz taxta"
            }

        label_en = names[idx]
        label = UZBEK_NAMES.get(label_en, label_en)
        conf  = float(preds[idx])

        all_probs = sorted(
            [{"label": UZBEK_NAMES.get(names[i], names[i]), "prob": float(preds[i])} for i in range(n_out)],
            key=lambda x: x["prob"], reverse=True
        )

        # Heatmap yaratish
        heatmap_b64 = None
        heatmap = make_gradcam_heatmap(arr, model, idx)
        if heatmap is not None:
            heatmap_b64 = overlay_heatmap(heatmap, img_bytes)

        return jsonify({
            "label": label,
            "confidence": conf,
            "all_probs": all_probs,
            "model_used": model_name,
            "heatmap": heatmap_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🪵 YOĞOCH NUQSON ANIQLOVCHI — WEB UI")
    print("="*55)
    print(f"  Yukangan modellar : {list(loaded_models.keys())}")
    print(f"  Klasslar ({len(CLASS_NAMES)} ta)  : {CLASS_NAMES}")
    print(f"  GPU               : {tf.config.list_physical_devices('GPU')}")
    print(f"\n  → http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
