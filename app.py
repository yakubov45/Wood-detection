import gradio as gr
import numpy as np
import tensorflow as tf
import gc
import os

# ── Model yuklash ──────────────────────────────────────────────────────────────
print("Model yuklanmoqda...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

model = tf.keras.models.load_model('best_finetuned_fixed.keras', compile=False)
# Warm-up
model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
print("✅ Model tayyor!")

CLASS_NAMES = ['Crack', 'Knot', 'Knot with Crack', 'Quartzite', 'Resin', 'Normal']

CLASS_INFO = {
    'Crack':          ('🔴', 'Yog\'och yuzasidagi yoriq — sinish xavfi bor'),
    'Knot':           ('🟠', 'Shox joyi — strukturaviy zaiflik'),
    'Knot with Crack':('🔴', 'Tugunli yoriq — jiddiy nuqson'),
    'Quartzite':      ('🟡', 'Mineral inklyuziya — qurilishda cheklangan'),
    'Resin':          ('🟡', 'Qatron oqishi — estetik nuqson'),
    'Normal':         ('🟢', 'Sog\'lom yog\'och — nuqson aniqlanmadi'),
}

# ── Bashorat funksiyasi ────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return None, "Rasm yuklanmadi"

    from PIL import Image as PILImage
    img = PILImage.fromarray(image).convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    preds = model.predict(arr[None], verbose=0)[0]
    gc.collect()

    # Label dict for Gradio Label component
    label_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    top_class = CLASS_NAMES[int(np.argmax(preds))]
    emoji, desc = CLASS_INFO[top_class]
    confidence = float(np.max(preds)) * 100
    info_text = f"{emoji} **{top_class}** — {confidence:.1f}% ishonch\n\n_{desc}_"

    return label_dict, info_text

# ── Gradio interfeysi ──────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    title="Wood Defect Classifier",
    css="""
        .gradio-container { max-width: 1000px !important; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # 🌲 Wood Defect Classifier
    **EfficientNet-B0** asosidagi yog'och nuqson aniqlovchi tizim.
    Rasmni yuklang yoki namuna rasmlardan birini tanlang.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Yog'och rasmi",
                type="numpy",
                height=300,
            )
            analyze_btn = gr.Button("🔍 Tahlil qilish", variant="primary", size="lg")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Kategoriyalar bo'yicha ehtimollik",
                num_top_classes=6,
            )
            info_output = gr.Markdown(label="Natija")

    # Namuna rasmlar
    examples_path = "examples"
    if os.path.exists(examples_path):
        example_files = [os.path.join(examples_path, f)
                         for f in os.listdir(examples_path)
                         if f.lower().endswith(('.jpg','.png','.jpeg'))]
        if example_files:
            gr.Examples(
                examples=example_files,
                inputs=image_input,
                label="Namuna rasmlar"
            )

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, info_output],
    )

    gr.Markdown("""
    ---
    *Model: EfficientNet-B0 · 6 kategoriya · TensorFlow 2.16*
    """)

if __name__ == "__main__":
    demo.launch()
