import os
import io
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

# ─── TFLite Model ─────────────────────────────────────────────────────────────
interpreter = None
input_details = None
output_details = None
model_error = None

def init_model():
    global interpreter, input_details, output_details, model_error
    try:
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.tflite')
        print(f"TFLite model: {model_path}")
        print(f"Fayl bor: {os.path.exists(model_path)}")

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ TFLite yuklandi! Input: {input_details[0]['shape']}")
    except Exception as e:
        model_error = str(e)
        print(f"❌ Model xatosi: {e}")
        traceback.print_exc()

print("=" * 50)
print("TFLite model yuklanmoqda...")
init_model()
print("=" * 50)

# ─── Kategoriyalar ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Crack',
    'Knot',
    'Knot with Crack',
    'Quartzite',
    'Resin',
    'Normal'
]

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok' if interpreter else 'error',
        'model_loaded': interpreter is not None,
        'error': model_error
    })

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'success': False, 'error': f'Model yuklanmadi: {model_error}'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Fayl topilmadi'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Fayl tanlanmadi'}), 400

    try:
        from PIL import Image

        # Preprocessing — model ichida Rescaling(1/255) bor
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)  # Raw [0-255]
        img_array = np.expand_dims(img_array, axis=0)

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

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
