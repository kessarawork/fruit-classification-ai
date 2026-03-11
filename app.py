import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json
import cv2
import os

st.set_page_config(page_title="Fruit AI Final Project", page_icon="🍎", layout="wide")

# --- โหลด Model ---
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('best_model.keras', compile=False)
        if os.path.exists('class_info.json'):
            with open('class_info.json', 'r', encoding='utf-8') as f:
                class_names = json.load(f)['classes']
        else:
            class_names = ["Apple", "Banana", "Grape", "Orange", "Pineapple"]
        return model, class_names
    except Exception as e:
        st.error(f"โหลดโมเดลไม่ได้: {e}")
        return None, None

model, class_names = load_resources()

# ฟังก์ชัน Saliency Map
def make_saliency_map(model, img_array, class_idx):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, img_tensor)
    if grads is None: return None
    
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    saliency = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-8)
    saliency = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    return heatmap

# ฟังก์ชัน Grad-CAM (Robust Fallback)
def make_gradcam_fallback(model, img_array, class_idx):
    # ใช้ Input Gradient มาทำ Heatmap แทน Grad-CAM กรณีเจาะ Layer ไม่ได้
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, img_tensor)
    
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    heatmap_smooth = cv2.GaussianBlur(dgrad_max_, (41, 41), 0)
    
    arr_min, arr_max = np.min(heatmap_smooth), np.max(heatmap_smooth)
    heatmap_norm = (heatmap_smooth - arr_min) / (arr_max - arr_min + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    heatmap_final = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    return heatmap_final, "Attention Map (Input-based)"

def make_gradcam_heatmap(model, img_array, class_idx):
    try:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            try:
                if len(layer.output_shape) == 4 and "input" not in layer.name:
                    last_conv_layer_name = layer.name
                    break
            except: continue
            
        if not last_conv_layer_name: raise ValueError("No Conv Layer")

        inputs_symbolic = model.input if not isinstance(model.input, list) else model.input[0]
        grad_model = tf.keras.models.Model(
            inputs=inputs_symbolic,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs_tf = tf.cast(img_array, tf.float32)
            conv_outputs, preds = grad_model(inputs_tf, training=False)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        heatmap_final = np.uint8(255 * heatmap.numpy())
        heatmap_final = cv2.applyColorMap(heatmap_final, cv2.COLORMAP_JET)
        return heatmap_final, last_conv_layer_name

    except Exception:
        return make_gradcam_fallback(model, img_array, class_idx)


# UI หลัก
st.title("🍎 Fruit Classification AI")
st.markdown("**Project Submission: AI Analysis**")

uploaded_file = st.sidebar.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"])

xai_method = st.sidebar.radio(
    "เลือกเทคนิค XAI:",
    ("Grad-CAM (Heatmap)", "Saliency Map (Pixel)")
)

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    # [FIXED HERE] แปลงรูปเป็น RGB (3 channels) เสมอ! ตัด Alpha ทิ้ง
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.image(image, caption='Original', use_container_width=True)
    
    target_size = (224, 224)
    image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # Preprocessing (0-255 Float32)
    img_array = np.asarray(image_resized).astype('float32')
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    try:
        preds = model.predict(img_batch, verbose=0)
        class_idx = np.argmax(preds[0])
        predicted_label = class_names[class_idx]
        confidence = np.max(preds[0])
        if confidence <= 1.0: confidence *= 100

        with col2:
            st.success(f"### ผลลัพธ์: {predicted_label}")
            st.metric("ความมั่นใจ", f"{confidence:.2f}%")
            
            probs = preds[0]
            if np.max(probs) > 1.0: probs = tf.nn.softmax(probs).numpy()
            st.bar_chart({name: float(p) for name, p in zip(class_names, probs)})

        # แสดงผล XAI
        st.write("---")
        st.subheader(f"🔍 AI Analysis: {xai_method}")
        
        try:
            heatmap = None
            
            if xai_method == "Grad-CAM (Heatmap)":
                heatmap, method_name = make_gradcam_heatmap(model, img_batch, class_idx)
                if "Input-based" in method_name:
                    st.caption("Mode: Robust Attention Map")
                else:
                    st.caption(f"Mode: Standard Grad-CAM ({method_name})")
                
            elif xai_method == "Saliency Map (Pixel)":
                heatmap = make_saliency_map(model, img_batch, class_idx)
                st.caption("Mode: Pixel Sensitivity")

            if heatmap is not None:
                original_cv = np.array(image_resized)
                original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
                
                heatmap = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
                overlay = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)
                
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="Heatmap", use_container_width=True)
                with img_col2:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay Result", use_container_width=True)
            else:
                st.error("ไม่สามารถสร้างภาพ Heatmap ได้")

        except Exception as e:
            st.error(f"XAI Error: {e}")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")