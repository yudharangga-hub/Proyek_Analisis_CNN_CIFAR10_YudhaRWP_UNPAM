import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import json
import keras_tuner as kt # Import KerasTuner

# --- Persiapan Awal ---
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('static'):
    os.makedirs('static')

print("Memuat dataset CIFAR-10...")
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
print("Dataset berhasil dimuat dan dinormalisasi.")

# =====================================================================
# BAGIAN 1: LATIH MODEL BASELINE (Tidak Berubah)
# =====================================================================
print("\n--- BAGIAN 1: Melatih Model Baseline ---")

# Bangun Arsitektur Model Baseline (Functional API)
inputs_base = layers.Input(shape=(32, 32, 3))
x_base = layers.Conv2D(32, (3, 3), activation='relu')(inputs_base)
x_base = layers.MaxPooling2D((2, 2))(x_base)
x_base = layers.Conv2D(64, (3, 3), activation='relu')(x_base)
x_base = layers.MaxPooling2D((2, 2))(x_base)
x_base = layers.Conv2D(64, (3, 3), activation='relu')(x_base)
x_base = layers.Flatten()(x_base)
x_base = layers.Dense(64, activation='relu')(x_base)
outputs_base = layers.Dense(10)(x_base)
model_baseline = models.Model(inputs=inputs_base, outputs=outputs_base)

# Kompilasi Model Baseline
model_baseline.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

# Latih Model Baseline
history_base = model_baseline.fit(train_images, train_labels, epochs=10, 
                                  validation_data=(test_images, test_labels), verbose=2)

# Simpan History Baseline ke JSON
print("Menyimpan history baseline ke JSON...")
baseline_history_clean = {k: [float(val) for val in v] for k, v in history_base.history.items()}
with open('static/baseline_history.json', 'w') as f:
    json.dump(baseline_history_clean, f)

# Simpan Model Baseline
model_baseline.save('models/baseline_cnn_cifar10.h5')
print("Model baseline telah disimpan.")


# =====================================================================
# BAGIAN 2: PENCARIAN MODEL OPTIMAL (BARU DENGAN KERASTUNER)
# =====================================================================
print("\n--- BAGIAN 2: Mencari Model Optimal (KerasTuner) ---")

# 1. Buat Fungsi Pembangun Model untuk Tuner
def build_optimal_model(hp):
    # Tentukan Hyperparameters (HP) yang akan dicari
    
    # HP untuk Augmentation
    hp_noise = hp.Float('noise', min_value=0.05, max_value=0.3, step=0.05)
    
    # HP untuk Arsitektur
    hp_filters_1 = hp.Int('filters_1', min_value=32, max_value=64, step=32)
    hp_filters_2 = hp.Int('filters_2', min_value=64, max_value=128, step=32)
    hp_dense_units = hp.Int('dense_units', min_value=64, max_value=128, step=32)
    
    # HP untuk Regularization
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    
    # HP untuk Kompilasi
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # --- Bangun Model ---
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Augmentation
    x = layers.GaussianNoise(hp_noise)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    
    # Blok Konvolusi 1
    x = layers.Conv2D(hp_filters_1, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x) # Tambahkan BatchNormalization
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Blok Konvolusi 2
    x = layers.Conv2D(hp_filters_2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Blok Fully Connected
    x = layers.Flatten()(x)
    x = layers.Dense(hp_dense_units, activation='relu')(x)
    x = layers.Dropout(hp_dropout)(x) # Tambahkan Dropout
    outputs = layers.Dense(10)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Kompilasi Model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

# 2. Inisialisasi Tuner
# Kita gunakan Hyperband: algoritma pencarian yang efisien
tuner = kt.Hyperband(build_optimal_model,
                     objective='val_accuracy', # Tujuan: maksimalkan akurasi validasi
                     max_epochs=20, # Epoch maksimum untuk satu model
                     factor=3,
                     directory='keras_tuner_dir', # Folder untuk menyimpan hasil
                     project_name='cifar10_optimization')

# Buat callback untuk menghentikan jika tidak ada peningkatan
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print("Memulai pencarian hyperparameter... (Ini akan memakan waktu lama)")
# 3. Jalankan Pencarian
tuner.search(train_images, train_labels,
             epochs=20,
             validation_data=(test_images, test_labels),
             callbacks=[stop_early])

# 4. Ambil Model Terbaik
print("\nPencarian selesai. Mengambil model terbaik...")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
optimal_model = tuner.get_best_models(num_models=1)[0]

print("--- Hyperparameter Optimal Ditemukan ---")
print(f"Noise Level: {best_hps.get('noise')}")
print(f"Filters L1: {best_hps.get('filters_1')}")
print(f"Filters L2: {best_hps.get('filters_2')}")
print(f"Dense Units: {best_hps.get('dense_units')}")
print(f"Dropout Rate: {best_hps.get('dropout')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print("------------------------------------------")


# =====================================================================
# BAGIAN 3: LATIH & SIMPAN MODEL OPTIMAL (BARU)
# =====================================================================
print("\n--- BAGIAN 3: Melatih Ulang Model Optimal untuk Final ---")

# Kita latih ulang model terbaik satu kali lagi untuk mendapatkan data history yang bersih
# (Meskipun tuner sudah melatihnya, ini adalah praktik yang baik)
history_optimal = optimal_model.fit(train_images, train_labels, epochs=20, 
                                    validation_data=(test_images, test_labels), verbose=2)

# Simpan History Optimal ke JSON
print("Menyimpan history optimal ke JSON...")
optimal_history_clean = {k: [float(val) for val in v] for k, v in history_optimal.history.items()}
with open('static/optimal_history.json', 'w') as f:
    json.dump(optimal_history_clean, f)

# Simpan Model Optimal
optimal_model.save('models/optimal_cnn_cifar10.h5')
print("Model optimal telah disimpan.")
print("\n--- SEMUA PROSES SELESAI ---")