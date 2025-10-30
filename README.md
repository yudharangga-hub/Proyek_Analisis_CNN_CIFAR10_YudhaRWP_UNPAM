# Proyek: From Noise to Clarity 🌫️➡️🔍
**Dashboard Analisis Ketahanan (Robustness) CNN pada CIFAR-10**

Repositori ini berisi kode sumber untuk aplikasi web interaktif yang dibuat dengan **Flask** dan **TensorFlow/Keras**. Aplikasi ini mendemonstrasikan dampak *noise* (gangguan visual) terhadap performa model Convolutional Neural Network (CNN) dan bagaimana **Data Augmentation** dapat digunakan untuk menciptakan model yang lebih "tahan banting".

**Diajukan untuk memenuhi Tugas Praktikum Mata Kuliah Advanced Computer Vision.**
**Nama:** Yudha Rangga Wulung Pura
**NIM:** 241012000151
**Institusi:** Magister Teknik Informatika - Universitas Pamulang

---

### 🚀 Fitur Utama

Aplikasi ini dibagi menjadi tiga bagian utama:

1.  **Analisis Interaktif:**
    * Membandingkan secara *real-time* performa model **Baseline** (rentan) vs. model **Robust** (tahan banting).
    * Menampilkan gambar hewan dari dataset CIFAR-10, versi aslinya, dan versi yang sudah "dirusak" oleh *Gaussian noise*.
    * Menunjukkan secara visual bagaimana model *Robust* tetap bisa menebak dengan benar sementara model *Baseline* gagal.

2.  **Hasil Pelatihan:**
    * Visualisasi plot *loss* dan *accuracy* dari kedua model selama proses pelatihan.
    * Menampilkan bukti visual terjadinya **overfitting** pada model *Baseline* dan bagaimana *data augmentation* mencegahnya di model *Robust*.

3.  **Visualisasi Fitur:**
    * Fitur canggih untuk "melihat ke dalam" pikiran model CNN.
    * Menampilkan *feature maps* (peta fitur) dari setiap lapisan konvolusi, menunjukkan bagaimana model belajar mendeteksi dari tepi, tekstur, hingga pola kompleks.

---

### 🔧 Cara Menjalankan Proyek Ini (Lokal)

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/NAMA-ANDA/Proyek_Analisis_CNN_CIFAR10.git](https://github.com/NAMA-ANDA/Proyek_Analisis_CNN_CIFAR10.git)
    cd Proyek_Analisis_CNN_CIFAR10
    ```

2.  **Buat virtual environment (Direkomendasikan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows: venv\Scripts\activate
    ```

3.  **Instal semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan aplikasi Flask:**
    ```bash
    flask run
    ```

5.  Buka browser Anda dan kunjungi `http://127.0.0.1:5000`

---

### 📂 Struktur Folder Proyek

```text
Proyek_CNN_CIFAR10/
│
├── app.py                   # Logika server Flask
├── main_project.py          # Skrip untuk melatih model dari awal
├── requirements.txt         # Daftar library Python
├── .gitignore               # File yang diabaikan Git
├── README.md                # Dokumentasi ini
│
├── models/
│   ├── baseline_cnn_cifar10.h5  # Model yang rentan
│   └── robust_cnn_cifar10.h5     # Model yang tahan banting
│
├── static/
│   ├── css/
│   │   └── style.css        # File CSS
│   └── images/
│       ├── logo_unpam.png   # Logo
│       └── ... (plot pelatihan)
│
└── templates/
    ├── base.html            # Template induk
    ├── index.html           # Halaman "Analisis Interaktif"
    ├── training_results.html  # Halaman "Hasil Pelatihan"
    └── feature_maps.html      # Halaman "Visualisasi Fitur"