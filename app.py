import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

app = Flask(__name__, static_folder='.')

# Model global o'zgaruvchi (bir marta yuklanadi)
model = None

def load_model():
    global model
    if model is None:
        import tensorflow as tf
        model_path = 'best_finetuned_fixed.keras'
        print(f"Model yuklanmoqda: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model tayyor!")
    return model

CLASS_NAMES = [
    'Crack (Yoriq)',
    'Knot (Tugun)',
    'Knot with Crack (Tugunli yoriq)',
    'Quartzite (Kvarsit)',
    'Resin (Qatron)',
    'Normal (Sog\'lom)'
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Fayl topilmadi'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fayl tanlanmadi'}), 400

    try:
        # Rasmni o'qish va preprocessing
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model bilan bashorat
        m = load_model()
        predictions = m.predict(img_array, verbose=0)[0]

        # Natijalarni tayyorlash
        results = []
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
