
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

# 🎨 طراحی ظاهر
st.set_page_config(page_title="Leaf Health Detection", page_icon="🌿", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0fff4;
    }
    .stButton>button {
        background-color: #38a169;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-size: 16px;
    }
    .stSuccess {
        font-size: 18px;
        font-weight: bold;
        color: #22543d;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🌿 سیستم آنلاین تشخیص سلامت برگ")
st.write("این اپلیکیشن با استفاده از مدل هوش مصنوعی، سلامت برگ‌ها را بررسی می‌کند.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("leaf_model.h5")
    return model

model = load_model()

# نمایش شکل ورودی مدل برای دیباگ
st.info(f"📐 شکل ورودی مدل: {model.input_shape}")

class_labels = ["apple_healthy", "apple_sick", "pear_healthy"]

def predict_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")

    # گرفتن سایز درست از مدل به‌صورت خودکار
    target_size = model.input_shape[1:3]   # مثلا (128,128) یا (224,224)
    img = img.resize(target_size)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,h,w,3)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    return class_labels[class_idx]

uploaded_file = st.file_uploader("📷 یک تصویر برگ آپلود کنید:", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📸 تصویر آپلود شده", use_container_width=True)
    with st.spinner("⏳ در حال پردازش..."):
        label = predict_image(uploaded_file)
    st.success(f"✅ نتیجه: برگ تشخیص داده شد به عنوان **{label}**")
