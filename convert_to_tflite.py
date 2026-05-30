"""
best_finetuned_fixed.keras → model.tflite (RAM tejash uchun)
"""
import tensorflow as tf
import os

print("Model yuklanmoqda...")
model = tf.keras.models.load_model('best_finetuned_fixed.keras')
print("Model yuklandi!")

print("TFLite ga o'tkazilmoqda (float32, quantizationsiz)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Quantization yo'q — op versiya muammosini oldini olish uchun
tflite_model = converter.convert()

output_path = 'model.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"✅ Tayyor! model.tflite: {size_mb:.1f} MB")
print("(Eski model: ~45MB, Yangi: ~11MB, RAM: ~50MB)")
