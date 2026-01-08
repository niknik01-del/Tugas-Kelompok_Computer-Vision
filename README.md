# Tugas-Kelompok_Computer-Vision
Folder ini berisi skrip dan data tugas kelompok 6_ computer vision
# Klasifikasi Awan Menggunakan ResNet CNN

![Header Image](images/header.png) <!-- Tambahkan gambar header jika ada -->

## 📋 Deskripsi Proyek
Proyek ini mengembangkan sistem klasifikasi citra awan berbasis **Convolutional Neural Network (CNN)** dengan arsitektur **Residual Network (ResNet)**. Sistem ini dapat mengklasifikasikan awan ke dalam 7 kelas berbeda dan diimplementasikan dalam sebuah web interface menggunakan Streamlit.

**Kelompok 6:**
- Niken Wahyuni (241012000138)
- Novana Sari (241012000127)
- Diah Ariefianty (241012000143)
- Lina Adrianti (241012000097)

**Mata Kuliah:** Computer Vision  
**Dosen:** Dr. Arya Adhyaksa Waskita, S.Si, M.Si  
**Institusi:** Program Pascasarjana Teknik Informatika, Universitas Pamulang

## 🎯 Domain Proyek
**Klasifikasi Citra Awan untuk Aplikasi Meteorologi**

### Latar Belakang
Awan merupakan salah satu elemen atmosfer yang memiliki peran penting dalam dinamika cuaca dan iklim. Karakteristik awan, seperti bentuk, ketinggian, dan struktur internal, sangat menentukan proses-proses atmosfer seperti pembentukan hujan, refleksi radiasi matahari, hingga perkembangan badai konvektif. Oleh karena itu, pengamatan dan identifikasi jenis awan secara akurat menjadi aspek fundamental dalam kegiatan meteorologi operasional, termasuk peringatan dini cuaca ekstrem dan keselamatan transportasi udara (Rahayu, Wibowo, & Handayani, 2022).
Di Indonesia, analisis awan memiliki urgensi yang lebih tinggi karena wilayah ini berada pada zona tropis yang aktif secara konvektif. Fenomena seperti hujan lebat tiba-tiba, thunderstorm, dan terbentuknya awan Cumulonimbus (Cb) sering terjadi dan berdampak signifikan terhadap aktivitas masyarakat serta sektor transportasi, terutama penerbangan. Identifikasi awan secara manual masih menjadi praktik umum dan sangat bergantung pada keterampilan pengamat, sehingga dapat menimbulkan subjektivitas serta inkonsistensi dalam proses klasifikasi (Putra, Sari, & Nugroho, 2021).
Seiring dengan meningkatnya ketersediaan citra cuaca resolusi tinggi—baik dari satelit geostasioner seperti Himawari-8 maupun dari kamera atmosfer permukaan—the volume data yang harus dianalisis semakin besar. Kondisi ini memerlukan sistem klasifikasi awan yang tidak hanya akurat, tetapi juga cepat dan mampu bekerja secara otomatis. Pemanfaatan deep learning, khususnya Convolutional Neural Network (CNN), telah menjadi salah satu pendekatan paling efektif dalam pengenalan pola visual karena kemampuannya dalam mengekstraksi fitur kompleks dari citra secara mandiri (Lestari & Sutrisno, 2023).
Berbagai penelitian di Indonesia menunjukkan bahwa CNN mampu memberikan kinerja yang unggul dalam pengolahan citra atmosfer dan cuaca. Abidin et al. (2023) berhasil mengimplementasikan CNN berbasis GoogLeNet untuk klasifikasi awan Cumulonimbus menggunakan citra Himawari-8 dengan akurasi mencapai 99%. Sementara Azis et al. (2025) mengembangkan metode segmentasi semantik awan Cumulonimbus yang juga menunjukkan performa akurasi tinggi di atas 99%. Penelitian lain yang memanfaatkan arsitektur CNN deep learning untuk pengenalan kondisi cuaca juga membuktikan efektivitas jaringan ini dalam menangani variasi tekstur dan pola langit yang kompleks (Tilasefana & Putra, 2023).
Selain itu, pendekatan CNN modern seperti Residual Network (ResNet) semakin banyak digunakan karena kemampuan residual learning-nya yang dapat mengatasi degradasi performa pada jaringan dalam. Penelitian oleh Lasniari et al. (2022) dan Thiodorus et al. (2021) menunjukkan bahwa ResNet memberikan akurasi tinggi pada tugas klasifikasi citra meskipun data latih terbatas, menjadikannya kandidat kuat untuk diterapkan pada klasifikasi citra awan multi-kelas. Di sisi aplikasi, integrasi model CNN pada platform berbasis web terbukti dapat meningkatkan keterjangkauan dan kemudahan penggunaan, seperti yang ditunjukkan dalam penelitian Tirtana et al. (2021) melalui aplikasi Herbify.
Melihat kebutuhan operasional terhadap sistem klasifikasi awan yang cepat, akurat, dan dapat diakses secara luas serta bukti empiris yang mendukung keberhasilan CNN dan ResNet, pengembangan model klasifikasi awan berbasis deep learning dan integrasinya dengan web interface sangat relevan dilakukan. Pendekatan ini diharapkan mampu mendukung proses analisis cuaca modern dan meningkatkan kualitas prediksi cuaca maupun keselamatan penerbangan di Indonesia.


## 📊 Business Understanding
### Problem Statements
1.	Bagaimana merancang dan membangun model klasifikasi citra awan multi-kelas berbasis ResNet CNN menggunakan PyTorch?
2.	Bagaimana kinerja model ResNet CNN yang dibangun dalam mengklasifikasikan citra awan ke dalam enam kelas (high cumuliform clouds, cumulus clouds, awan tinggi, stratocumulus clouds, cumulonimbus clouds, dan clear sky) ditinjau dari metrik akurasi, presisi, recall, dan F1-score serta confusion matrix?
3.	Bagaimana merancang dan mengimplementasikan prototipe web interface yang memanfaatkan model ResNet CNN tersebut untuk melakukan klasifikasi citra awan secara interaktif sehingga hasil model dapat diakses melalui browser?


### Goals
1.	Mengembangkan model klasifikasi citra awan multi-kelas berbasis ResNet CNN menggunakan PyTorch.
2.	Menganalisis kinerja model ResNet CNN dalam mengklasifikasikan citra awan ke dalam enam kelas dengan menggunakan metrik akurasi, presisi, recall, F1-score, serta confusion matrix yang diekstraksi dari hasil pengujian.
3.	Merancang dan mengimplementasikan prototipe web interface yang terintegrasi dengan model ResNet CNN sehingga pengguna dapat mengunggah citra awan dan memperoleh hasil klasifikasi secara otomatis melalui web interface


## 🗂️ Data Understanding
### Dataset
- **Sumber:** Dataset citra awan berwarna (RGB)
- **Jumlah Kelas:** 7 kelas awan
- **Kelas:** High Cumuliform Clouds, Cumulus Clouds, Cirriform Clouds, Stratiform Clouds, Stratocumulus Clouds, Cumulonimbus Clouds, Clear Sky
- **Struktur:** Data dibagi menjadi training set dan validation set

### Statistik Dataset
| Kelas | Jumlah Sampel | Contoh Gambar |
|-------|---------------|---------------|
| High Cumuliform Clouds | 77 | ![High Cumuliform](images/high_cumuliform_sample.jpg) |
| Cumulus Clouds | 64 | ![Cumulus](images/cumulus_sample.jpg) |
| Cirriform Clouds | 11 | ![Cirriform](images/cirriform_sample.jpg) |
| Stratiform Clouds | 120 | ![Stratiform](images/stratiform_sample.jpg) |
| Stratocumulus Clouds | 103 | ![Stratocumulus](images/stratocumulus_sample.jpg) |
| Cumulonimbus Clouds | 40 | ![Cumulonimbus](images/cumulonimbus_sample.jpg) |
| Clear Sky | 71 | ![Clear Sky](images/clear_sky_sample.jpg) |

## ⚙️ Data Preparation
### Preprocessing
1. **Resizing:** Citra diubah ukuran menjadi 384×384 piksel
2. **Normalisasi:** Menggunakan mean dan std dari ImageNet
3. **Augmentasi:**
   - Random Horizontal Flip (p=0.5)
   - Random Vertical Flip (p=0.5)
   - Random Rotation (18 derajat)

### Transformasi
```python
train_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=18),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
