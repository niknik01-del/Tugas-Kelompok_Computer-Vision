# ☁️ Klasifikasi Citra Awan Menggunakan Custom ResNet CNN

**Project UAS – Advanced Computer Vision**  
Program Studi Teknik Informatika S2  
Universitas Pamulang  
Tahun 2025

---

## 📄 Informasi Umum

Penelitian ini mengembangkan sistem **klasifikasi citra awan multi-kelas** berbasis **Deep Learning** menggunakan **Convolutional Neural Network (CNN)** dengan arsitektur **Custom Residual Network (ResNet)**. Model dilatih menggunakan citra awan RGB resolusi tinggi dan diimplementasikan ke dalam **web interface berbasis Streamlit** untuk mendukung inferensi secara real-time.

---

## 👨‍👩‍👧‍👦 Tim Penyusun

**Kelompok 6**

| Nama | NIM |
|-----|-----|
| Niken Wahyuni | 241012000138 |
| Novana Sari | 241012000127 |
| Diah Ariefianty | 241012000143 |
| Lina Adrianti | 241012000097 |

**Dosen Pengampu:**  
Dr. Arya Adhyaksa Waskita, S.Si., M.Si.

---

## 🎯 Tujuan Penelitian

- Mengembangkan model klasifikasi citra awan multi-kelas berbasis **ResNet CNN**
- Menganalisis kinerja model menggunakan **Accuracy, Precision, Recall, F1-score**, dan **Confusion Matrix**
- Mengimplementasikan model ke dalam **web interface interaktif** berbasis Streamlit

---

## 🧠 Kelas Awan

Model mengenali **7 kelas awan**:

1. High Cumuliform Clouds  
2. Cumulus Clouds  
3. Cirriform Clouds  
4. Stratiform Clouds  
5. Stratocumulus Clouds  
6. Cumulonimbus Clouds  
7. Clear Sky  

---

## 🗂 Dataset & Pra-pemrosesan

- Format data: **Citra RGB**
- Struktur: **ImageFolder (PyTorch)**
- Pembagian data: **Training** dan **Validation**

### Preprocessing & Augmentasi
Augmentasi hanya diterapkan pada **training set**:

- Resize → **384 × 384**
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation (±18°)
- Normalisasi ImageNet

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 🏗 Arsitektur Model

Model menggunakan **Custom CNN bergaya ResNet** dengan residual block.

| Blok | Konfigurasi |
|-----|------------|
| Conv Block 1 | Conv2D (3 → 64), BatchNorm, ReLU |
| Conv Block 2 | Conv2D (64 → 128) + MaxPool |
| Residual Block 1 | ConvBlock(128→128) + Skip |
| Conv Block 3 | Conv2D (128 → 256) + MaxPool |
| Conv Block 4 | Conv2D (256 → 512) + MaxPool |
| Residual Block 2 | ConvBlock(512→512) + Skip |
| Output | AdaptiveAvgPool → FC (7 kelas) |

---

## ⚙️ Konfigurasi Pelatihan

| Parameter | Nilai |
|--------|------|
| Epoch | 40 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Scheduler | OneCycleLR |
| Loss Function | CrossEntropyLoss |
| Gradient Clipping | 0.15 |

---

## 📊 Hasil Evaluasi

- **Akurasi:** 89.30%
- **Macro F1-score:** 85.99%

---

## 🌐 Web Interface

Aplikasi web dikembangkan menggunakan **Streamlit** untuk inferensi real-time.

---

## 📁 Struktur Repository

```
├── dataset/
├── models_web_h5/
├── training/
│   └── Cloud_klasi.py
├── app/
│   └── streamlit_app.py
├── README.md
└── requirements.txt
```

---

## 🚀 Cara Menjalankan

```
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 📄 Laporan Lengkap

👉 [Laporan Klasifikasi Awan (DOCX/PDF)](./Laporan_Klasifikasi_Awan.pdf)

---

## 📝 Lisensi

Proyek ini dibuat untuk **keperluan akademik (UAS)**.
