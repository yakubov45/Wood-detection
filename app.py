import os
import io
import json
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

# Model global o'zgaruvchi
model = None
model_error = None

def load_model():
    global model, model_error
    if model is not None:
        return model, None
    if model_error is not None:
        return None, model_error
    try:
        import tensorflow as tf
        print("TensorFlow versiyasi:", tf.__version__)
        model_path = os.path.join(os.path.dirname(__file__), 'best_finetuned_fixed.keras')
        print(f"Model yuklanmoqda: {model_path}")
        print(f"Fayl mavjudmi: {os.path.exists(model_path)}")
        model = tf.keras.models.load_model(model_path)
        print("Model muvaffaqiyatli yuklandi!")
        return model, None
    except Exception as e:
        model_error = str(e)
        traceback.print_exc()
        return None, model_error

CLASS_NAMES = [
    'Crack (Yoriq)',
    'Knot (Tugun)',
    'Knot with Crack (Tugunli yoriq)',
    'Quartzite (Kvarsit)',
    'Resin (Qatron)',
    "Normal (Sog'lom)"
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    """Server holati tekshirish"""
    m, err = load_model()
    return jsonify({
        'status': 'ok' if m else 'model_error',
        'model_loaded': m is not None,
        'error': err
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Fayl borligini tekshirish
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Fayl topilmadi'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Fayl tanlanmadi'}), 400

    try:
        from PIL import Image

        # Modalni yuklash
        m, err = load_model()
        if m is None:
            return jsonify({'success': False, 'error': f'Model yuklanmadi: {err}'}), 500

        # Rasmni o'qish
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Bashorat
        predictions = m.predict(img_array, verbose=0)[0]

        # Natijalar
        results = []
        for name, prob in zip(CLASS_NAMES, predictions):
            results.append({
                'class': name,
                'probability': float(prob),
                'percentage': round(float(prob) * 100, 1)
            })

        results.sort(key=lambda x: x['probability'], reverse=True)
        top = results[0]

        return jsonify({
            'success': True,
            'prediction': top['class'],
            'confidence': top['percentage'],
            'all_results': results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Barcha boshqa xatolar uchun JSON javob
@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
