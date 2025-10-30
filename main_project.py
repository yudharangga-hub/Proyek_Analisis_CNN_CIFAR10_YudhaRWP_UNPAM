### Proyek From Noise to Clarity: Langkah 1 - Model Baseline ###

# 1.1. Import Library
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Persiapan Awal ---

# Buat folder 'models' dan 'results' jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')
    print("Folder 'models' telah dibuat.")

if not os.path.exists('results'):
    os.makedirs('results')
    print("Folder 'results' telah dibuat.")


# 1.2. Persiapan Dataset
print("Memuat dataset CIFAR-10...")
# Muat dan bagi dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalisasi nilai piksel dari [0, 255] ke [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

print(f"Data latih: {train_images.shape}")
print(f"Data tes: {test_images.shape}")
print("Dataset berhasil dimuat dan dinormalisasi.")


# 1.3. Bangun Arsitektur Model CNN
# (Berdasarkan slide "Implementasi CNN Sederhana dengan Keras")
print("Membangun model CNN...")
model = models.Sequential()

# Layer konvolusi pertama
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Layer konvolusi kedua
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Layer konvolusi ketiga
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten 3D feature map ke 1D vector
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) # 10 output untuk 10 kelas (tanpa aktivasi softmax)

# Tampilkan ringkasan arsitektur model
model.summary()


# 1.4. Kompilasi Model
print("Mengkompilasi model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Model berhasil dikompilasi.")


# 1.5. Latih Model (Training)
print("Memulai pelatihan model (10 epoch)...")
# Latih model pada data training, validasi menggunakan data testing
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
print("Pelatihan selesai.")


# 1.6. Evaluasi dan Simpan Model
print("Mengevaluasi model pada data tes (bersih)...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nAkurasi pada data tes (bersih): {test_acc*100:.2f}%')

# Simpan model yang sudah dilatih ke folder 'models'
model.save('models/baseline_cnn_cifar10.h5')
print("Model baseline telah disimpan di 'models/baseline_cnn_cifar10.h5'")


# 1.7. Visualisasi Hasil Pelatihan (Plot)
print("Menyimpan plot hasil pelatihan...")
plt.figure(figsize=(12, 4))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

# Simpan plot ke folder 'results'
# Simpan plot ke folder 'static/images' agar bisa diakses web
if not os.path.exists('static/images'):
    os.makedirs('static/images')
plt.savefig('static/images/baseline_training_plot.png')
print("Plot pelatihan telah disimpan di 'results/baseline_training_plot.png'")

print("\n--- Langkah 1 Selesai ---")

### Proyek From Noise to Clarity: Langkah 2 - Uji Coba Gangguan ###

print("\n--- Memulai Langkah 2: Uji Coba Gangguan ---")

# 2.1. Muat Model Baseline
# Kita tidak perlu melatih ulang, cukup muat model yang sudah disimpan
from tensorflow.keras.models import load_model

print("Memuat model baseline dari 'models/baseline_cnn_cifar10.h5'...")
model = load_model('models/baseline_cnn_cifar10.h5')

# (Kita masih memiliki test_images dan test_labels dari Langkah 1)
# Jika Anda menjalankan ini di file terpisah, Anda perlu memuat ulang data tes.

# 2.2. Fungsi untuk Menambah Gangguan (Noise)
# Kita akan membuat fungsi untuk "Gaussian Noise"
# Ini seperti "semut" pada sinyal TV yang buruk

def tambah_gaussian_noise(images, noise_factor=0.1):
    print(f"Menambahkan Gaussian noise dengan faktor {noise_factor}...")
    # Buat salinan gambar agar tidak merusak data asli
    noisy_images = images.copy() 
    
    # Hasilkan noise acak dari distribusi normal
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=noisy_images.shape)
    
    # Tambahkan noise ke gambar
    noisy_images = noisy_images + noise
    
    # Pastikan nilai piksel tetap di antara 0 dan 1
    noisy_images = np.clip(noisy_images, 0., 1.)
    
    return noisy_images

# 2.3. Buat Dataset Tes yang "Rusak"
test_images_noisy = tambah_gaussian_noise(test_images, noise_factor=0.2) 
# Anda bisa bereksperimen dengan noise_factor ini (misal: 0.1, 0.3, 0.5)

# 2.4. Evaluasi Model pada Data "Rusak"
print("Mengevaluasi model pada data tes yang 'rusak' (noisy)...")
test_loss_noisy, test_acc_noisy = model.evaluate(test_images_noisy, test_labels, verbose=2)

print(f'\nAkurasi pada data tes (BERSIH): {test_acc*100:.2f}%')
print(f'Akurasi pada data tes (NOISY): {test_acc_noisy*100:.2f}%')

# 2.5. Visualisasi Perbandingan (Bukti)
print("Menyimpan perbandingan gambar di 'results/noise_comparison.png'...")

# Tentukan nama kelas CIFAR-10
class_names = ['pesawat', 'mobil', 'burung', 'kucing', 'rusa', 
               'anjing', 'katak', 'kuda', 'kapal', 'truk']

plt.figure(figsize=(15, 6))
for i in range(5):
    # Ambil gambar acak
    idx = np.random.randint(0, test_images.shape[0])
    
    # Gambar Asli
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[idx])
    plt.title(f"Asli: {class_names[test_labels[idx][0]]}")
    plt.axis('off')
    
    # Gambar Rusak (Noisy)
    plt.subplot(2, 5, i + 6)
    plt.imshow(test_images_noisy[idx])
    
    # Dapatkan prediksi model
    pred_noisy = model.predict(test_images_noisy[idx:idx+1])
    pred_class = class_names[np.argmax(pred_noisy)]
    
    plt.title(f"Noisy (Pred: {pred_class})")
    plt.axis('off')

plt.savefig('results/noise_comparison.png')
print("Perbandingan gambar telah disimpan.")
print("\n--- Langkah 2 Selesai ---")

### Proyek From Noise to Clarity: Langkah 3 - Melatih Model Robust ###

print("\n--- Memulai Langkah 3: Melatih Model Robust dengan Data Augmentation ---")

# (Kita asumsikan train_images dan train_labels masih ada dari Langkah 1)
# Jika tidak, Anda harus memuat ulang datanya

# 3.1. Definisikan Layer Data Augmentation
# Kita akan gunakan layer Keras untuk ini
data_augmentation = models.Sequential(
    [
        # Menambahkan noise, SAMA SEPERTI GANGGUAN KITA
        layers.GaussianNoise(0.2), 
        
        # Tambahkan augmentasi lain yang umum untuk CIFAR-10
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ],
    name="data_augmentation",
)

# 3.2. Bangun Arsitektur Model Robust
# Kita akan membuat model BARU, tapi kita letakkan augmentation di dalamnya
print("Membangun model robust (Augmentation + CNN)...")

# Kita butuh Input layer terpisah
inputs = tf.keras.Input(shape=(32, 32, 3))

# 1. Terapkan augmentasi
x = data_augmentation(inputs) 

# 2. Terapkan sisa arsitektur CNN (SAMA SEPERTI baseline)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)

# Buat model baru
model_robust = models.Model(inputs, outputs)

model_robust.summary()

# 3.3. Kompilasi Model Robust
print("Mengkompilasi model robust...")
model_robust.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

# 3.4. Latih Model Robust
print("Memulai pelatihan model robust...")
# Catatan: Karena augmentasi membuat tugas lebih sulit,
# kita latih sedikit lebih lama, misal 20 epoch.
history_robust = model_robust.fit(train_images, train_labels, epochs=20, 
                                validation_data=(test_images, test_labels))

print("Pelatihan model robust selesai.")

# 3.5. Simpan Model Robust
model_robust.save('models/robust_cnn_cifar10.h5')
print("Model robust telah disimpan di 'models/robust_cnn_cifar10.h5'")

# 3.6. Simpan Plot Pelatihan Robust
print("Menyimpan plot hasil pelatihan robust...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_robust.history['accuracy'], label='Training Accuracy')
plt.plot(history_robust.history['val_accuracy'], label='Validation Accuracy')
plt.title('Robust Model Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_robust.history['loss'], label='Training Loss')
plt.plot(history_robust.history['val_loss'], label='Validation Loss')
plt.title('Robust Model Loss')
plt.legend()
# Simpan plot ke folder 'static/images'
plt.savefig('static/images/robust_training_plot.png')
print("Plot pelatihan robust telah disimpan.")

print("\n--- Langkah 3 Selesai ---")

### Proyek From Noise to Clarity: Langkah 4 - Analisis Perbandingan Akhir ###

print("\n--- Memulai Langkah 4: Analisis Perbandingan Akhir ---")

# 4.1. Muat KEDUA model
print("Memuat kedua model (baseline dan robust)...")
model_baseline = load_model('models/baseline_cnn_cifar10.h5')
model_robust = load_model('models/robust_cnn_cifar10.h5')

# 4.2. Siapkan data tes (Bersih dan Rusak)
# (Kita sudah punya test_images dan test_labels)
# Kita buat LAGI data yang noisy
test_images_noisy = tambah_gaussian_noise(test_images, noise_factor=0.2) 

# 4.3. Evaluasi Model Baseline
print("\nMengevaluasi Model BASELINE...")
_, acc_baseline_clean = model_baseline.evaluate(test_images, test_labels, verbose=0)
_, acc_baseline_noisy = model_baseline.evaluate(test_images_noisy, test_labels, verbose=0)

# 4.4. Evaluasi Model Robust
print("Mengevaluasi Model ROBUST...")
_, acc_robust_clean = model_robust.evaluate(test_images, test_labels, verbose=0)
_, acc_robust_noisy = model_robust.evaluate(test_images_noisy, test_labels, verbose=0)

# 4.5. Tampilkan Hasil Akhir
print("\n--- HASIL AKHIR PROYEK ---")
print(f"Akurasi Baseline (Data BERSIH): {acc_baseline_clean*100:.2f}%")
print(f"Akurasi Baseline (Data NOISY):  {acc_baseline_noisy*100:.2f}%")
print("---------------------------------")
print(f"Akurasi Robust (Data BERSIH): {acc_robust_clean*100:.2f}%")
print(f"Akurasi Robust (Data NOISY):  {acc_robust_noisy*100:.2f}%")
print("---------------------------------")

penurunan_baseline = acc_baseline_clean - acc_baseline_noisy
penurunan_robust = acc_robust_clean - acc_robust_noisy

print(f"Penurunan Akurasi Baseline: {penurunan_baseline*100:.2f}%")
print(f"Penurunan Akurasi Robust:   {penurunan_robust*100:.2f}%")
print("\n--- Proyek Selesai ---")