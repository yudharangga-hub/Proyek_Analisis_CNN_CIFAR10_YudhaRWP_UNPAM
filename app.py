import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import cifar10
from flask import Flask, render_template, request, jsonify # Pastikan jsonify ada
import random
import io
import base64
from PIL import Image, ImageEnhance
import matplotlib.cm as cm

# --- 1. Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ini-rahasia-jangan-disebar'

# --- 2. PENGATURAN MODEL ---
print("Memuat model... Harap tunggu.")
# Pastikan Anda sudah mengganti nama file 'optimal' menjadi 'robust'
MODEL_BASELINE = load_model('models/baseline_cnn_cifar10.h5')
MODEL_ROBUST = load_model('models/robust_cnn_cifar10.h5') 
print("Model berhasil dimuat.")

# Temukan nama layer konvolusi terakhir secara otomatis
LAST_CONV_LAYER_NAME_BASELINE = [layer.name for layer in MODEL_BASELINE.layers if 'conv2d' in layer.name][-1]
LAST_CONV_LAYER_NAME_ROBUST = [layer.name for layer in MODEL_ROBUST.layers if 'conv2d' in layer.name][-1]

print(f"Layer Grad-CAM Baseline: {LAST_CONV_LAYER_NAME_BASELINE}")
print(f"Layer Grad-CAM Robust: {LAST_CONV_LAYER_NAME_ROBUST}")

# --- 3. PENGATURAN DATA ---
CLASS_NAMES = ['pesawat', 'mobil', 'burung', 'kucing', 'rusa', 
               'anjing', 'katak', 'kuda', 'kapal', 'truk']
ZOO_CLASSES_IDX = [2, 3, 4, 5, 6, 7] # burung, kucing, rusa, anjing, katak, kuda

print("Memuat dan memfilter data ZOO...")
(_, _), (test_images, test_labels) = cifar10.load_data()
zoo_indices = np.where(np.isin(test_labels.flatten(), ZOO_CLASSES_IDX))[0]
ZOO_IMAGES = test_images[zoo_indices] # 0-255
ZOO_LABELS = test_labels[zoo_indices]
ZOO_IMAGES_NORMALIZED = ZOO_IMAGES.astype('float32') / 255
print(f"Data ZOO siap. Total gambar hewan: {len(ZOO_IMAGES)}")


# --- 4. FUNGSI BANTUAN ---

def tambah_gaussian_noise(image_normalized, noise_factor=0.2):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image_normalized.shape)
    noisy_image = np.clip(image_normalized + noise, 0., 1.)
    return noisy_image

def encode_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def get_prediction(model, image_normalized):
    img_batch = np.expand_dims(image_normalized, axis=0)
    pred_logits = model.predict(img_batch)
    pred_idx = np.argmax(pred_logits)
    pred_name = CLASS_NAMES[pred_idx]
    return pred_name, pred_idx

def get_random_zoo_image_with_index():
    """Mengembalikan index asli dari dataset CIFAR-10"""
    idx = random.randint(0, len(ZOO_IMAGES) - 1)
    img_display = ZOO_IMAGES[idx] # 0-255
    img_predict = ZOO_IMAGES_NORMALIZED[idx] # 0-1
    label_idx = ZOO_LABELS[idx][0]
    label_name = CLASS_NAMES[label_idx]
    original_idx = zoo_indices[idx]
    return img_display, img_predict, label_name, original_idx

def get_zoo_image_by_index(original_idx):
    """Mengambil gambar berdasarkan index aslinya"""
    zoo_idx = np.where(zoo_indices == original_idx)[0][0]
    img_display = ZOO_IMAGES[zoo_idx] # 0-255
    img_predict = ZOO_IMAGES_NORMALIZED[zoo_idx] # 0-1
    label_idx = ZOO_LABELS[zoo_idx][0]
    label_name = CLASS_NAMES[label_idx]
    return img_display, img_predict, label_name

def get_grad_cam_heatmap(model, last_conv_layer_name, img_array_normalized, pred_index):
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    img_batch = np.expand_dims(img_array_normalized, axis=0)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_batch)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    if tf.math.reduce_max(heatmap) > 0:
        heatmap /= tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    jet = cm.get_cmap("jet")
    jet_colors = jet(heatmap)[:, :, :3]
    jet_heatmap = (jet_colors * 255).astype(np.uint8)
    return jet_heatmap

def create_grad_cam_visuals(model, last_conv_layer_name, img_display, img_predict):
    pred_name, pred_idx = get_prediction(model, img_predict)
    heatmap_array = get_grad_cam_heatmap(model, last_conv_layer_name, img_predict, pred_idx)
    original_pil = Image.fromarray(img_display)
    heatmap_pil = Image.fromarray(heatmap_array).resize(original_pil.size, Image.BILINEAR)
    superimposed_pil = Image.blend(original_pil, heatmap_pil, alpha=0.4)
    heatmap_b64 = encode_image_to_base64(heatmap_pil.resize((256, 256), Image.NEAREST))
    superimposed_b64 = encode_image_to_base64(superimposed_pil.resize((256, 256), Image.NEAREST))
    return pred_name, heatmap_b64, superimposed_b64


# --- 5. RUTE HALAMAN WEB (CONTROLLERS) ---

@app.route('/')
def home():
    """Halaman Utama: Analisis Interaktif"""
    img_display, img_predict, true_label, image_index = get_random_zoo_image_with_index()
    default_noise = 0.2
    noisy_image_predict = tambah_gaussian_noise(img_predict, noise_factor=default_noise)
    noisy_image_display = (noisy_image_predict * 255).astype('uint8')
    pred_base_clean, _ = get_prediction(MODEL_BASELINE, img_predict)
    pred_base_noisy, _ = get_prediction(MODEL_BASELINE, noisy_image_predict)
    pred_robust_clean, _ = get_prediction(MODEL_ROBUST, img_predict)
    pred_robust_noisy, _ = get_prediction(MODEL_ROBUST, noisy_image_predict)
    clean_image_b64 = encode_image_to_base64(Image.fromarray(img_display).resize((256, 256), Image.NEAREST))
    noisy_image_b64 = encode_image_to_base64(Image.fromarray(noisy_image_display).resize((256, 256), Image.NEAREST))

    return render_template('index.html',
                           true_label=true_label,
                           image_index=image_index,
                           default_noise=default_noise,
                           clean_image_b64=clean_image_b64,
                           noisy_image_b64=noisy_image_b64,
                           pred_base_clean=pred_base_clean,
                           pred_robust_clean=pred_robust_clean,
                           pred_base_noisy=pred_base_noisy,
                           pred_robust_noisy=pred_robust_noisy
                           )

@app.route('/update_noise', methods=['POST'])
def update_noise():
    """Rute AJAX untuk slider"""
    data = request.json
    noise_level = float(data['noise_level'])
    image_index = int(data['image_index'])
    
    img_display, img_predict, true_label = get_zoo_image_by_index(image_index)
    
    noisy_image_predict = tambah_gaussian_noise(img_predict, noise_factor=noise_level)
    noisy_image_display = (noisy_image_predict * 255).astype('uint8')
    
    pred_base_noisy, _ = get_prediction(MODEL_BASELINE, noisy_image_predict)
    pred_robust_noisy, _ = get_prediction(MODEL_ROBUST, noisy_image_predict)
    
    noisy_image_b64 = encode_image_to_base64(Image.fromarray(noisy_image_display).resize((256, 256), Image.NEAREST))
    
    return jsonify({
        'noisy_image_b64': noisy_image_b64,
        'pred_base_noisy': pred_base_noisy,
        'pred_robust_noisy': pred_robust_noisy,
        'true_label': true_label
    })


@app.route('/training_results')
def training_results():
    """Halaman Tab 2: Hasil Pelatihan"""
    return render_template('training_results.html')


@app.route('/feature_maps')
def feature_maps():
    """Halaman Tab 3: Perbandingan Grad-CAM"""
    img_display, img_predict, true_label, _ = get_random_zoo_image_with_index()
    
    pred_base, heat_base_b64, super_base_b64 = create_grad_cam_visuals(
        MODEL_BASELINE, LAST_CONV_LAYER_NAME_BASELINE, img_display, img_predict
    )
    pred_robust, heat_robust_b64, super_robust_b64 = create_grad_cam_visuals(
        MODEL_ROBUST, LAST_CONV_LAYER_NAME_ROBUST, img_display, img_predict
    )
    original_b64 = encode_image_to_base64(Image.fromarray(img_display).resize((256, 256), Image.NEAREST))

    # --- INI ADALAH BARIS YANG BENAR ---
    # Perhatikan bagaimana setiap variabel memiliki 'nama' (keyword)
    return render_template('feature_maps.html',
                           true_label=true_label,
                           original_image_b64=original_b64,
                           
                           pred_base=pred_base,
                           heat_base_b64=heat_base_b64,
                           super_base_b64=super_base_b64,
                           
                           pred_robust=pred_robust,
                           heat_robust_b64=heat_robust_b64,
                           super_robust_b64=super_robust_b64
                           )

@app.route('/optimization_analysis')
def optimization_analysis():
    """Halaman Tab 4: Menampilkan Hasil Tuning"""
    return render_template('optimization.html')


# --- 6. MENJALANKAN SERVER ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)