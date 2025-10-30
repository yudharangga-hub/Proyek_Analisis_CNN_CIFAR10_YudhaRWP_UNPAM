import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import cifar10
from flask import Flask, render_template, request
import random
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ini-rahasia-jangan-disebar' # Dibutuhkan untuk session

# --- 2. PENGATURAN MODEL (HANYA SEKALI SAAT SERVER NYALA) ---
print("Memuat model... Harap tunggu.")
MODEL_BASELINE = load_model('models/baseline_cnn_cifar10.h5')
MODEL_ROBUST = load_model('models/robust_cnn_cifar10.h5')
print("Model berhasil dimuat.")

# Ekstrak layer konvolusi dari Model Robust untuk visualisasi
# (Sesuai slide "Visualisasi Feature Maps" [cite: 170])
layer_outputs = [layer.output for layer in MODEL_ROBUST.layers if 'conv2d' in layer.name]
layer_names = [layer.name for layer in MODEL_ROBUST.layers if 'conv2d' in layer.name]
ACTIVATION_MODEL = Model(inputs=MODEL_ROBUST.input, outputs=layer_outputs)
print(f"Model aktivasi dibuat dengan {len(layer_names)} layer.")

# --- 3. PENGATURAN DATA (HANYA SEKALI SAAT SERVER NYALA) ---
CLASS_NAMES = ['pesawat', 'mobil', 'burung', 'kucing', 'rusa', 
               'anjing', 'katak', 'kuda', 'kapal', 'truk']
ZOO_CLASSES_IDX = [2, 3, 4, 5, 6, 7] # burung, kucing, rusa, anjing, katak, kuda

print("Memuat dan memfilter data ZOO...")
(_, _), (test_images, test_labels) = cifar10.load_data()
zoo_indices = np.where(np.isin(test_labels.flatten(), ZOO_CLASSES_IDX))[0]
ZOO_IMAGES = test_images[zoo_indices] # 0-255 (untuk display)
ZOO_LABELS = test_labels[zoo_indices]
ZOO_IMAGES_NORMALIZED = ZOO_IMAGES.astype('float32') / 255
print(f"Data ZOO siap. Total gambar hewan: {len(ZOO_IMAGES)}")


# --- 4. FUNGSI BANTUAN ---

def tambah_gaussian_noise(image_normalized, noise_factor=0.2):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image_normalized.shape)
    noisy_image = np.clip(image_normalized + noise, 0., 1.)
    return noisy_image

def encode_image_to_base64(image_array_0_255, size=(256, 256)):
    """Mengubah array NumPy (0-255) menjadi string Base64 untuk HTML"""
    img = Image.fromarray(image_array_0_255.astype('uint8'))
    if size[0] == size[1]: # Jika persegi, gunakan pixelated
        img = img.resize(size, Image.NEAREST)
    else: # Jika tidak persegi (feature map), gunakan bilinear
        img = img.resize(size, Image.BILINEAR)
        
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def get_prediction(model, image_normalized):
    """Mendapat prediksi nama kelas dari model"""
    img_batch = np.expand_dims(image_normalized, axis=0)
    pred = model.predict(img_batch)
    pred_idx = np.argmax(pred)
    return CLASS_NAMES[pred_idx]

def get_random_zoo_image():
    """Mengambil satu set data gambar ZOO secara acak"""
    idx = random.randint(0, len(ZOO_IMAGES) - 1)
    img_display = ZOO_IMAGES[idx] # 0-255
    img_predict = ZOO_IMAGES_NORMALIZED[idx] # 0-1
    label_idx = ZOO_LABELS[idx][0]
    label_name = CLASS_NAMES[label_idx]
    return img_display, img_predict, label_name


# --- 5. RUTE HALAMAN WEB (CONTROLLERS) ---

@app.route('/')
def home():
    """Halaman Utama: Analisis Interaktif"""
    img_display, img_predict, true_label = get_random_zoo_image()

    # Buat versi "rusak"
    noisy_image_predict = tambah_gaussian_noise(img_predict, noise_factor=0.2)
    noisy_image_display = (noisy_image_predict * 255).astype('uint8')

    # Dapatkan semua 4 prediksi
    pred_base_clean = get_prediction(MODEL_BASELINE, img_predict)
    pred_base_noisy = get_prediction(MODEL_BASELINE, noisy_image_predict)
    pred_robust_clean = get_prediction(MODEL_ROBUST, img_predict)
    pred_robust_noisy = get_prediction(MODEL_ROBUST, noisy_image_predict)

    # Encode gambar ke Base64 untuk HTML
    clean_image_b64 = encode_image_to_base64(img_display, size=(256, 256))
    noisy_image_b64 = encode_image_to_base64(noisy_image_display, size=(256, 256))

    return render_template('index.html',
                           true_label=true_label,
                           clean_image_b64=clean_image_b64,
                           noisy_image_b64=noisy_image_b64,
                           pred_base_clean=pred_base_clean,
                           pred_base_noisy=pred_base_noisy,
                           pred_robust_clean=pred_robust_clean,
                           pred_robust_noisy=pred_robust_noisy
                           )

@app.route('/training_results')
def training_results():
    """Halaman Tab 2: Menampilkan Plot Hasil Pelatihan"""
    # File HTML akan memuat gambar dari /static/images/
    return render_template('training_results.html')


@app.route('/feature_maps')
def feature_maps():
    """Halaman Tab 3: Visualisasi Aktivasi Layer CNN"""
    img_display, img_predict, true_label = get_random_zoo_image()

    # 1. Dapatkan aktivasi dari model
    img_batch = np.expand_dims(img_predict, axis=0)
    layer_activations = ACTIVATION_MODEL.predict(img_batch)
    
    # 2. Proses aktivasi untuk dikirim ke HTML
    activations_for_html = []
    
    for layer_name, layer_activation in zip(layer_names, layer_activations):
        num_features = layer_activation.shape[-1]
        layer_data = {"name": f"Lapisan: {layer_name} ({num_features} fitur)", "maps": []}
        
        # Ambil 16 feature map pertama saja (agar tidak terlalu berat)
        for i in range(min(num_features, 16)):
            feature_map = layer_activation[0, :, :, i]
            
            # Normalisasi untuk visualisasi
            feature_map -= feature_map.mean()
            feature_map /= feature_map.std()
            feature_map *= 64
            feature_map += 128
            feature_map = np.clip(feature_map, 0, 255)
            
            # Encode ke Base64
            map_b64 = encode_image_to_base64(feature_map, size=(100, 100))
            layer_data["maps"].append(map_b64)
            
        activations_for_html.append(layer_data)

    # 3. Encode gambar input asli
    input_image_b64 = encode_image_to_base64(img_display, size=(256, 256))

    return render_template('feature_maps.html',
                           true_label=true_label,
                           input_image_b64=input_image_b64,
                           activations=activations_for_html
                           )


# --- 6. MENJALANKAN SERVER ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)