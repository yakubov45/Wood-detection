import os
import io
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

# ─── Model startup da yuklanadi ───────────────────────────────────────────────
model = None
model_error = None

def init_model():
    global model, model_error
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_finetuned_fixed.keras')
        print(f"Model path: {model_path}")
        print(f"Fayl bor: {os.path.exists(model_path)}")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model muvaffaqiyatli yuklandi!")
    except Exception as e:
        model_error = str(e)
        print(f"❌ Model xatosi: {e}")
        traceback.print_exc()

# Startup da yukla
print("=" * 50)
print("Model yuklanmoqda (startup)...")
init_model()
print("=" * 50)

# ─── Kategoriyalar ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Crack (Yoriq)',
    'Knot (Tugun)',
    'Knot with Crack (Tugunli yoriq)',
    'Quartzite (Kvarsit)',
    'Resin (Qatron)',
    "Normal (Sog'lom)"
]

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok' if model else 'error',
        'model_loaded': model is not None,
        'error': model_error
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': f'Model yuklanmadi: {model_error}'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Fayl topilmadi'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Fayl tanlanmadi'}), 400

    try:
        from PIL import Image

        # Rasm preprocessing — model ichida Rescaling(1/255) bor, /255 qilmaymiz
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)  # Raw [0-255]
        img_array = np.expand_dims(img_array, axis=0)

        # Bashorat
        predictions = model.predict(img_array, verbose=0)[0]

        results = []
        for name, prob in zip(CLASS_NAMES, predictions):
            results.append({
                'class': name,
                'probability': float(prob),
                'percentage': round(float(prob) * 100, 1)
            })
        results.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'prediction': results[0]['class'],
            'confidence': results[0]['percentage'],
            'all_results': results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
