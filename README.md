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

## Dosen Pengampu  
Dr. Arya Adhyaksa Waskita, S.Si., M.Si.
---


# 1. Pendahuluan

### 1.1 Latar Belakang

Awan merupakan salah satu elemen atmosfer yang memiliki peran penting dalam dinamika
cuaca dan iklim. Karakteristik awan, seperti bentuk, ketinggian, dan struktur internal,
sangat menentukan berbagai proses atmosfer, termasuk pembentukan hujan, refleksi
radiasi matahari, hingga perkembangan badai konvektif. Oleh karena itu, pengamatan dan
identifikasi jenis awan secara akurat menjadi aspek fundamental dalam kegiatan
meteorologi operasional, khususnya untuk peringatan dini cuaca ekstrem dan keselamatan
transportasi udara (Rahayu, Wibowo, & Handayani, 2022).

Di Indonesia, urgensi analisis awan menjadi lebih tinggi karena wilayah ini berada
pada zona tropis yang aktif secara konvektif. Fenomena hujan lebat tiba-tiba,
*thunderstorm*, serta pembentukan awan **Cumulonimbus (Cb)** sering terjadi dan
memberikan dampak signifikan terhadap aktivitas masyarakat dan sektor transportasi,
terutama penerbangan. Hingga saat ini, identifikasi awan secara manual masih banyak
digunakan dan sangat bergantung pada keterampilan pengamat, sehingga berpotensi
menimbulkan subjektivitas dan inkonsistensi dalam proses klasifikasi
(Putra, Sari, & Nugroho, 2021).

Seiring meningkatnya ketersediaan citra cuaca resolusi tinggi, baik dari satelit
geostasioner seperti **Himawari-8** maupun dari kamera atmosfer permukaan, volume data
yang harus dianalisis semakin besar. Kondisi ini menuntut adanya sistem klasifikasi
awan yang tidak hanya akurat, tetapi juga cepat dan mampu bekerja secara otomatis.
Pemanfaatan *deep learning*, khususnya **Convolutional Neural Network (CNN)**, telah
menjadi salah satu pendekatan paling efektif dalam pengenalan pola visual karena
kemampuannya mengekstraksi fitur kompleks dari citra secara mandiri
(Lestari & Sutrisno, 2023).

Berbagai penelitian di Indonesia menunjukkan bahwa CNN mampu memberikan kinerja yang
unggul dalam pengolahan citra atmosfer dan cuaca. Abidin et al. (2023) berhasil
mengimplementasikan CNN berbasis GoogLeNet untuk klasifikasi awan Cumulonimbus
menggunakan citra Himawari-8 dengan akurasi mencapai 99%. Sementara itu, Azis et al.
(2025) mengembangkan metode segmentasi semantik awan Cumulonimbus yang juga
menunjukkan performa akurasi tinggi di atas 99%. Penelitian lain yang memanfaatkan
arsitektur CNN untuk pengenalan kondisi cuaca turut membuktikan efektivitas jaringan
ini dalam menangani variasi tekstur dan pola langit yang kompleks
(Tilasefana & Putra, 2023).

Pendekatan CNN modern seperti **Residual Network (ResNet)** semakin banyak digunakan
karena kemampuan *residual learning* yang dapat mengatasi degradasi performa pada
jaringan yang dalam. Penelitian oleh Lasniari et al. (2022) dan Thiodorus et al. (2021)
menunjukkan bahwa ResNet mampu menghasilkan akurasi tinggi meskipun jumlah data latih
terbatas, sehingga menjadikannya kandidat kuat untuk klasifikasi citra awan multi-kelas.
Dari sisi aplikasi, integrasi model CNN ke dalam platform berbasis web juga terbukti
meningkatkan keterjangkauan dan kemudahan penggunaan, sebagaimana ditunjukkan oleh
Tirtana et al. (2021) melalui pengembangan aplikasi berbasis web.

Berdasarkan kebutuhan operasional akan sistem klasifikasi awan yang cepat, akurat,
dan mudah diakses, serta dukungan bukti empiris terkait keberhasilan CNN dan ResNet,
pengembangan model klasifikasi awan berbasis *deep learning* yang terintegrasi dengan
web interface menjadi sangat relevan. Pendekatan ini diharapkan dapat mendukung proses
analisis cuaca modern dan meningkatkan kualitas prediksi cuaca serta keselamatan
penerbangan di Indonesia.

---

### 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini
dirumuskan sebagai berikut:

1. Bagaimana merancang dan membangun model klasifikasi citra awan multi-kelas berbasis
   **ResNet CNN** menggunakan framework **PyTorch**?
2. Bagaimana kinerja model ResNet CNN dalam mengklasifikasikan citra awan ke dalam
   beberapa kelas awan, ditinjau dari metrik **akurasi, precision, recall, F1-score**,
   serta **confusion matrix**?
3. Bagaimana merancang dan mengimplementasikan prototipe **web interface** yang
   memanfaatkan model ResNet CNN untuk melakukan klasifikasi citra awan secara
   interaktif sehingga hasil prediksi dapat diakses melalui browser?



---
## 📖 Latar Belakang & Masalah
Identifikasi jenis awan sangat krusial untuk meteorologi, keselamatan penerbangan, dan prediksi cuaca ekstrem (seperti awan Cumulonimbus yang berbahaya). Namun, pengamatan manual sering kali subjektif dan tidak konsisten.
Proyek ini bertujuan untuk membangun model otomatis yang akurat menggunakan ResNet CNN untuk mengatasi masalah vanishing gradient pada jaringan dalam, serta menyediakan antarmuka yang mudah digunakan bagi forecaster pemula maupun masyarakat umum.

## 🎯 Tujuan Penelitian

- Mengembangkan model klasifikasi citra awan multi-kelas berbasis **ResNet CNN**
- Menganalisis kinerja model menggunakan **Accuracy, Precision, Recall, F1-score**, dan **Confusion Matrix**
- Mengimplementasikan model ke dalam **web interface interaktif** berbasis Streamlit


### 1.3 Manfaat Penelitian

Penelitian ini diharapkan dapat memberikan manfaat baik secara teoritis maupun
praktis sebagai berikut.

#### 1.3.1 Manfaat Teoritis
Manfaat teoritis dari penelitian ini meliputi:
- Menambah khazanah penelitian di bidang pengolahan citra awan dan cuaca di Indonesia
  dengan pendekatan *deep learning*, khususnya melalui pemanfaatan arsitektur
  **ResNet CNN** untuk klasifikasi citra awan multi-kelas.
- Memberikan contoh implementasi arsitektur **CNN bergaya ResNet kustom**
  (*residual block*) pada citra awan yang dikembangkan menggunakan **PyTorch**,
  termasuk perancangan alur pelatihan dan evaluasi model.
- Menunjukkan proses persiapan model klasifikasi citra yang telah dilatih untuk
  keperluan *deployment* melalui ekspor bobot (*weights*) dan label kelas, sehingga
  dapat menjadi referensi bagi penelitian sejenis yang berorientasi pada implementasi
  aplikasi.

---

#### 1.3.2 Manfaat Praktis
Manfaat praktis yang diharapkan dari penelitian ini antara lain:
- Menyediakan prototipe sistem klasifikasi citra awan berbasis web yang dapat
  dijadikan dasar pengembangan lebih lanjut di lingkungan operasional **BMKG** atau
  instansi terkait, guna mendukung analisis kondisi awan secara lebih cepat dan
  konsisten.
- Membantu *forecaster* atau analis cuaca pemula dalam mengenali tipe awan tertentu,
  khususnya awan konvektif berbahaya, melalui sistem cerdas yang memberikan prediksi
  kelas secara otomatis.
- Menjadi contoh penerapan *end-to-end pipeline*, mulai dari pemrosesan data,
  pelatihan model *deep learning*, evaluasi kinerja, hingga integrasi ke *web
  interface*, yang dapat diadaptasi untuk aplikasi lain di bidang meteorologi maupun
  pengolahan citra secara umum.

---

### 1.4 Batasan Masalah

Agar penelitian lebih terarah dan fokus, ditetapkan beberapa batasan masalah sebagai
berikut:
1. Data citra awan yang digunakan berupa **citra statis (gambar tunggal)**, bukan
   deret waktu (*time-series*) maupun data video.
2. Klasifikasi dibatasi pada **enam kelas awan** sebagaimana didefinisikan dalam skrip
   `Cloud_klasi.py`, yaitu:
   - High Cumuliform Clouds  
   - Cumulus Clouds  
   - Cirriform Clouds (awan tinggi)  
   - Stratocumulus Clouds  
   - Cumulonimbus Clouds  
   - Clear Sky
3. Ukuran citra masukan pada model ResNet CNN dibatasi menjadi **384 × 384 piksel**
   dengan tiga kanal warna (**RGB**). Teknik augmentasi yang digunakan meliputi
   rotasi, *horizontal flip*, *vertical flip*, serta normalisasi.
4. Arsitektur model yang digunakan adalah **CNN bergaya ResNet kustom**
   (`CNN_NeuralNet`).
5. *Web interface* yang dikembangkan bersifat **prototipe**, dibatasi pada fungsi
   utama untuk:
   - Mengunggah citra awan  
   - Memanggil model ResNet CNN yang telah dilatih melalui berkas bobot dan label  
   - Menampilkan hasil klasifikasi  

   Integrasi penuh dengan sistem operasional BMKG dan alur data satelit real-time
   berada di luar ruang lingkup penelitian ini.

---
# LANDASAN TEORI
## Convolutional Neural Network (CNN)
Convolutional Neural Network (CNN) merupakan salah satu arsitektur deep learning yang dirancang khusus untuk mengolah data berbentuk grid, seperti citra digital. CNN bekerja dengan mengekstraksi fitur visual melalui operasi convolution, activation, pooling, dan fully connected layer. Setiap lapisan bertanggung jawab untuk mempelajari pola berbeda, mulai dari pola sederhana (tepi/garis) hingga pola kompleks (tekstur dan bentuk objek) (Tilasefana & Putra, 2023).
Keunggulan CNN terletak pada kemampuannya mengekstraksi local features dan menghasilkan representasi citra yang efisien tanpa memerlukan rekayasa fitur manual. CNN terbukti efektif untuk tugas klasifikasi citra, termasuk citra atmosfer, karena mampu mengenali pola awan yang memiliki bentuk dan tekstur kompleks. Penelitian Tuna dan Kristianto (2024) menegaskan bahwa CNN dapat mencapai akurasi tinggi dalam klasifikasi citra cuaca dengan memanfaatkan struktur hierarkis pada citra langit.
Berbagai penelitian di Indonesia mendukung efektivitas CNN dalam analisis citra cuaca dan awan. Misalnya, Abidin et al. (2023) menunjukkan bahwa CNN GoogLeNet mampu mengklasifikasikan pembentukan awan Cumulonimbus dengan akurasi hingga 99%, membuktikan bahwa CNN dapat digunakan untuk mendeteksi fenomena atmosfer berbahaya.

## Residual Network (ResNet)
Residual Network (ResNet) diperkenalkan oleh He et al. (2015) untuk mengatasi masalah vanishing gradient yang umum terjadi pada jaringan berlapis-lapis. Inti dari arsitektur ResNet adalah residual block, yaitu mekanisme shortcut connection yang memungkinkan model melewati satu atau beberapa lapisan sehingga gradien dapat mengalir tanpa hambatan selama proses backpropagation. Hal ini membuat ResNet dapat dilatih dengan kedalaman puluhan hingga ratusan lapisan tanpa mengalami penurunan performa.
Penggunaan ResNet di Indonesia telah terbukti efektif pada berbagai domain klasifikasi citra. Lasniari et al. (2022) menggunakan ResNet-50 untuk klasifikasi citra daging sapi dan babi, dan model berhasil mencapai akurasi tinggi meskipun perbedaan antar objek sangat halus. Rifqi (2022) juga menunjukkan bahwa transfer learning menggunakan ResNet dapat meningkatkan performa klasifikasi pada citra X-Ray. Dalam konteks citra atmosfer, ResNet memiliki keunggulan karena fitur awan sering memiliki variasi tekstur yang kompleks. ResNet dapat menangkap pola mendalam dan halus pada struktur awan, seperti perbedaan cirrus, cumulus, stratocumulus, hingga cumulonimbus.

## Klasifikasi Awan
Awan diklasifikasikan berdasarkan bentuk, ketinggian, dan proses pembentukannya menjadi beberapa kelompok dasar, seperti cirriform, cumuliform, stratiform, stratocumulus, dan cumulonimbus. Pengklasifikasian awan memiliki peran penting dalam meteorologi karena tipe awan dapat menjadi indikator kondisi cuaca tertentu. Misalnya, awan cirrus menandakan kondisi stabil di atmosfer bagian atas, sementara cumulonimbus menandakan aktivitas konvektif kuat yang berpotensi menimbulkan badai, petir, hujan deras, dan turbulensi penerbangan (Rahayu et al., 2022).
Sejumlah penelitian Indonesia menekankan pentingnya klasifikasi awan berbasis citra. Abidin et al. (2023) menunjukkan bahwa identifikasi awan Cumulonimbus pada citra satelit dapat digunakan sebagai indikator awal potensi badai. Sementara itu, Azis et al. (2025) mengembangkan model segmentasi awan Cb yang mampu memetakan area awan berbahaya dengan akurasi tinggi.
Pengenalan awan berbasis citra juga penting dalam sistem peringatan dini cuaca. Penelitian Putra et al. (2021) menekankan bahwa klasifikasi citra langit dapat digunakan untuk membantu prakiraan cuaca otomatis, terutama pada kondisi cerah, berawan, mendung, dan hujan. Pendekatan semacam ini semakin relevan dengan meningkatnya ketersediaan citra beresolusi tinggi dari satelit dan kamera atmosfer.
Dengan adanya perkembangan deep learning, klasifikasi awan kini banyak menggunakan CNN karena mampu mengidentifikasi pola tekstur awan yang kompleks. Ke depannya, integrasi model CNN ke web interface akan semakin meningkatkan aksesibilitas informasi bagi forecaster dan pengguna lainnya.

## 🧠 Kelas Awan

Model mengenali **7 kelas awan**:

1. High Cumuliform Clouds  
2. Cumulus Clouds  
3. Cirriform Clouds  
4. Stratiform Clouds  
5. Stratocumulus Clouds  
6. Cumulonimbus Clouds  
7. Clear Sky  


# METODOLOGI
## Desain Penelitian
Penelitian ini menggunakan pendekatan kuantitatif eksperimental (experimental research design) yang berfokus pada pengembangan dan evaluasi model deep learning untuk melakukan klasifikasi citra awan. Model yang dikembangkan berbasis Convolutional Neural Network (CNN) dengan arsitektur Residual Network (ResNet) yang dirancang secara khusus (custom) sesuai dengan kebutuhan klasifikasi multi-kelas.
Prosedur penelitian meliputi beberapa tahapan utama, yaitu:
1.	Pengumpulan dataset citra awan dan penataan struktur direktori data.
2.	Pra-pemrosesan citra melalui normalisasi dan augmentasi untuk meningkatkan representativitas data.
3.	Perancangan arsitektur CNN–ResNet yang efisien untuk mengekstraksi pola visual awan.
4.	Pelatihan model (training) menggunakan optimizer AdamW dan penjadwal laju pembelajaran OneCycleLR.
5.	Evaluasi kinerja model menggunakan metrik kuantitatif standar klasifikasi.
6.	Penyimpanan model dan integrasi ke dalam sistem antarmuka web

## Dataset Penelitian
## Sumber dan Karakteristik Dataset
Dataset yang digunakan dalam penelitian ini terdiri atas **citra awan berwarna (RGB)**
yang dikelompokkan ke dalam **tujuh kelas awan**, yaitu:
- High Cumuliform Clouds
- Cumulus Clouds
- Stratocumulus Clouds
- Cirriform Clouds
- Cumulonimbus Clouds
- Stratiform Clouds
- Clear Sky

Struktur dataset disusun ke dalam dua direktori utama, yaitu **training** dan
**testing**, di mana masing-masing direktori memiliki subfolder yang merepresentasikan
kelas awan. Pendekatan struktur folder ini mengikuti standar **ImageFolder** pada
pustaka **PyTorch**, sehingga mempermudah proses pemanggilan dan pemuatan data secara
otomatis selama proses pelatihan dan evaluasi model.

Dataset citra awan dapat diakses melalui Google Drive pada tautan berikut:  
🔗 **Dataset Cloud Image Classification**  
https://drive.google.com/drive/folders/1oas9aTRA2fpiYRSELMFiZXWN0Dk1Y29u?usp=drive_link


## Pembagian Dataset
Dataset dipisahkan secara manual ke dalam dua subset utama, yaitu:
- **Training set**, digunakan untuk membangun dan melatih model klasifikasi.
- **Validation/Testing set**, digunakan untuk mengukur performa model secara objektif
  terhadap data yang belum pernah dilihat selama proses pelatihan.

Pemilahan manual ini memungkinkan proporsi dataset dipertahankan sesuai kebutuhan eksperimen.

## Pra-pemrosesan dan Augmentasi Citra
Proses pra-pemrosesan (preprocessing) yang dilakukan meliputi:
1.	Resizing citra menjadi 384 × 384 piksel untuk menyeragamkan resolusi.
2.	Normalisasi menggunakan nilai mean dan standard deviation dari dataset ImageNet:
 
3.	Data Augmentation mencakup :
- **RandomHorizontalFlip**
- **RandomVerticalFlip**
- **RandomRotation** dengan sudut rotasi sebesar **±18°**

Penerapan augmentasi ini bertujuan untuk memperkaya distribusi data pelatihan serta
meningkatkan kemampuan generalisasi model terhadap variasi orientasi dan struktur
awan. Pendekatan ini sejalan dengan rekomendasi **Lasniari et al. (2022)** dan
**Putra et al. (2021)** yang menekankan pentingnya diversifikasi data dalam tugas
klasifikasi citra atmosfer.

Gambar berikut menunjukkan contoh hasil augmentasi data pada citra awan yang digunakan
dalam proses pelatihan model.


<img src="assets/Augmented_training images.jpeg" width="700">

## Perancangan Model
## Convolutional Neural Network (CNN)
CNN berfungsi sebagai fondasi utama dalam proses ekstraksi fitur citra awan. Lapisan
konvolusi melakukan pemindaian pola piksel untuk mengekstraksi fitur tingkat rendah,
seperti tepi, hingga fitur tingkat tinggi yang merepresentasikan struktur dan pola
awan.

Model CNN yang digunakan dalam penelitian ini terdiri dari beberapa komponen utama,
yaitu:
- **Convolutional Layer** dengan kernel berukuran **3×3** untuk mengekstraksi fitur
  spasial.
- **Batch Normalization (BatchNorm2d)** untuk menstabilkan distribusi aktivasi dan
  mempercepat proses pelatihan.
- **Fungsi aktivasi ReLU** untuk memperkenalkan sifat non-linear pada jaringan.
- **MaxPooling** dengan ukuran **4×4** untuk mengurangi dimensi fitur dan kompleksitas
  komputasi.
- **Fully Connected Layer (Linear)** untuk memetakan fitur hasil ekstraksi ke dalam
  kelas awan yang diklasifikasikan.

Desain arsitektur ini mengikuti konfigurasi yang umum digunakan dalam model
klasifikasi citra cuaca dan atmosfer, sebagaimana ditunjukkan dalam penelitian
**Tilasefana & Putra (2023)**.


## Residual Network (ResNet) Versi Custom
Model menggunakan arsitektur ResNet yang dimodifikasi secara khusus berdasarkan skrip penelitian. ResNet menggunakan residual block, yaitu mekanisme koneksi langsung (shortcut connection) yang memungkinkan jaringan melewati satu atau lebih lapisan, sehingga meminimalkan hilangnya informasi selama proses pembelajaran.
Dalam skrip, residual block dibangun menggunakan struktur berikut:
  **Dua ConvBlock berturut-turut** untuk mengekstraksi fitur spasial secara mendalam.
- **Penjumlahan berbasis identity mapping** antara input dan output blok konvolusi
  untuk menjaga kontinuitas informasi.
- **Fungsi aktivasi ReLU** yang diterapkan setelah proses penjumlahan untuk
  memperkenalkan non-linearitas.

Arsitektur lengkap mencakup:
## 🏗 Arsitektur Model

| Bagian Model | Konfigurasi |
|------------|-------------|
| Block 1 | ConvBlock (3 → 16) + MaxPooling 4× |
| Block 2 | ConvBlock (16 → 64) + MaxPooling 4× |
| Block 3 | ConvBlock (64 → 128) + Residual Block (128 → 128) |
| Block 4 | ConvBlock (128 → 256) + MaxPooling 4× |
| Block 5 | ConvBlock (256 → 512) + Residual Block (512 → 512) |
| Output | Adaptive Average Pooling → Fully Connected Output |


## Pengaturan Hyperparameter
●	Loss Function: CrossEntropyLoss
●	Optimizer: AdamW (lebih stabil terhadap weight decay)
●	Learning Rate (LR): 0.001
●	Scheduler: OneCycleLR
●	Epoch: 40
●	Batch size: 32
●	Gradient Clipping: 0.15 (untuk menjaga stabilitas pelatihan)
Pemilihan AdamW dan OneCycleLR terbukti mengoptimalkan proses konvergensi sebagaimana dijelaskan dalam berbagai studi deep learning terkini.

## Mekanisme Pelatihan
Tahapan pelatihan meliputi:
1.	Forward pass: citra diproses melalui jaringan untuk menghasilkan prediksi.
2.	Perhitungan loss: menggunakan cross-entropy.
3.	Backward pass: gradien dihitung menggunakan backpropagation.
4.	Update parameter: dilakukan oleh AdamW.
5.	Penyesuaian learning rate: OneCycleLR mengatur LR dinamika tinggi di awal dan menurun di akhir.
Seluruh nilai training loss dan validation loss direkam dalam sebuah berkas JSON sebagai arsip pelacakan performa.

## Evaluasi Model
Evaluasi model dilakukan secara sistematis menggunakan metrik berikut:
1. Akurasi
Mengukur persentase prediksi yang benar dibandingkan seluruh sampel.
2. Precision, Recall, dan F1-score
Digunakan untuk mengukur performa per kelas, terutama penting saat dataset tidak seimbang.
Misalnya, kelas awan Cumulonimbus biasanya memiliki lebih sedikit sampel, sehingga precision dan recall menjadi indikator yang lebih informatif.
3. Confusion Matrix
Confusion matrix divisualisasikan menggunakan ConfusionMatrixDisplay dan seaborn heatmap.
Diagram *confusion matrix* memberikan informasi penting terkait perilaku dan
karakteristik kesalahan model klasifikasi awan. Analisis diagram ini digunakan untuk
memahami pola prediksi model secara lebih mendalam, khususnya pada setiap kelas awan.

Informasi utama yang diperoleh dari diagram tersebut meliputi:
- **Kelas awan yang paling sering tertukar**, sehingga dapat diidentifikasi pasangan
  kelas dengan kemiripan visual tinggi.
- **Proporsi prediksi benar pada setiap kelas**, yang mencerminkan tingkat
  keandalan model dalam mengenali pola visual awan tertentu.
- **Karakteristik kesalahan model**, termasuk kecenderungan bias terhadap kelas
  tertentu dan dampak ketidakseimbangan data.

Pendekatan evaluasi menggunakan *confusion matrix* ini mengikuti praktik yang umum
digunakan dalam penelitian klasifikasi dan segmentasi awan, sebagaimana ditunjukkan
dalam studi **Abidin et al. (2023)** dan **Azis et al. (2025)**.


## Implementasi Model ke Web Interface
Antarmuka web (web interface) untuk klasifikasi awan ini memanfaatkan Streamlit sebagai kerangka kerja rapid prototyping untuk menyajikan laporan dan hasil inferensi model secara interaktif dan real-time kepada pengguna, menghilangkan kebutuhan akan pengembangan front-end manual (HTML/CSS/JavaScript). Streamlit menciptakan sinergi antara kemampuan pemodelan Machine Learning (PyTorch CNN) dan tampilan web yang disederhanakan

## Arsitektur Deployment
Model inferensi dilakukan secara Server-Side untuk memastikan hasil prediksi konsisten dan aman digunakan pada perangkat apa pun. Pendekatan deployment ini, di mana model CNN disinergikan dengan aplikasi web dalam konteks identifikasi objek visual, mengikuti pola yang ditunjukkan dalam penelitian Tirtana et al. (2021).
Streamlit bertindak sebagai mesin renderer yang menerjemahkan script Python Anda menjadi elemen web interaktif. Ini memungkinkan sinergi yang efisien antara model yang kompleks (seperti CNN) dan antarmuka web yang dapat diakses dari perangkat apa pun..
Berbeda dari pola full-stack tradisional (Flask/FastAPI + HTML/CSS/JS), arsitektur deployment Streamlit ini menggabungkan frontend dan backend dalam satu lapisan Python:
- Backend (Logika Model): Streamlit bertindak sebagai server side di mana model PyTorch CNN diinisialisasi dan dieksekusi
- Frontend (Tampilan Pengguna): Streamlit secara otomatis merender script Python menjadi elemen web interaktif (tombol, metrik, grafik)
- Desain Kustom: Untuk mencapai estetika Neumorphism (3D) dan tema Candy Land, CSS kustom diinjeksi (custom CSS injection) langsung ke Streamlit untuk menimpa gaya bawaan dan memberikan nuansa multi-panel pastel yang unik


## Tahapan Pra-Deployment (Server-Side)

Pada tahap pra-deployment, seluruh aset dan logika yang dibutuhkan untuk proses
pelaporan dan prediksi disiapkan di sisi server (komputer tempat aplikasi Streamlit
dijalankan). Pendekatan ini memastikan konsistensi hasil, keamanan pemrosesan, dan
efisiensi komputasi, khususnya ketika model dijalankan menggunakan GPU.

### 1. Pelatihan Model dan Server-Side Inferensi
Model klasifikasi awan berbasis **CNN (Custom ResNet)** dilatih menggunakan framework
**PyTorch**. Proses inferensi, yaitu prediksi terhadap citra baru, sepenuhnya dilakukan
di sisi server tempat Streamlit berjalan. Pendekatan *server-side inference* ini
dipilih untuk:

- Menjaga konsistensi hasil prediksi  
- Meningkatkan keamanan model dan data  
- Mengoptimalkan pemanfaatan sumber daya komputasi (CPU/GPU)

---

### 2. Proses Inferensi (Server-Side)
Seluruh proses inferensi dilakukan secara **server-side**. Ketika pengguna mengunggah
gambar dan menekan tombol prediksi, alur yang terjadi adalah sebagai berikut:

1. Gambar input dikirimkan dari browser pengguna ke server Streamlit.
2. Server menjalankan gambar tersebut melalui model CNN yang telah dimuat ke memori
   menggunakan mekanisme *caching* (`@st.cache_resource` dan `@st.cache_data`).
3. Hasil prediksi yang konsisten dan aman dihasilkan di server.
4. Output prediksi kemudian dikirim kembali ke browser untuk ditampilkan kepada pengguna.

Pendekatan ini memastikan bahwa seluruh proses komputasi inti tetap berada di sisi
server dan tidak bergantung pada kemampuan perangkat pengguna.

---

### 3. Penyimpanan Aset Statis untuk Deployment
Folder hasil dari proses pelatihan berisi seluruh artefak statis yang diperlukan
untuk menjalankan aplikasi web (Streamlit) serta mendokumentasikan kinerja model CNN
dalam format yang mudah dibaca dan divisualisasikan. Aset-aset tersebut meliputi:

- **`cloud_resnet_state_dict.h5`**  
  Berisi bobot (*weights*) final dari model CNN Custom ResNet setelah 40 epoch pelatihan.
  File ini merupakan representasi matematis akhir model dan digunakan oleh Streamlit
  untuk menjalankan proses inferensi.

- **`cloud_resnet_full.h5`**  
  Menyimpan model secara lengkap, termasuk arsitektur dan bobotnya. File ini berfungsi
  sebagai opsi cadangan (*backup*) atau digunakan pada skenario deployment yang
  memerlukan pemuatan model secara utuh.

- **`cloud_labels.json`**  
  Berisi daftar nama kelas awan yang dikenali oleh model, seperti *cumulus clouds* dan
  *clear sky*. Informasi ini digunakan untuk memastikan jumlah kelas (7 kelas) yang
  menjadi fokus klasifikasi serta ditampilkan pada bagian konfigurasi dan laporan.

- **`resnet_Model.pth`**  
  Merupakan bobot model sementara (*checkpoint*) yang dihasilkan pada akhir fase
  pelatihan di skrip `tugas_cv.py`. File ini kemudian dimuat ulang untuk membentuk
  aset deployment final (`cloud_resnet_state_dict.h5`).

- **`cloud_metrics.json`**  
  Menyimpan metrik evaluasi akhir model pada *validation set*, termasuk nilai Akurasi,
  Macro Average F1-Score, serta metrik rinci (Precision, Recall, dan F1-Score) untuk
  setiap kelas. File ini juga berisi data mentah untuk visualisasi *Confusion Matrix*.

- **`training_history.json`**  
  Berisi data historis pelatihan berupa nilai *Loss* dan *Accuracy* pada setiap epoch.
  Data ini digunakan untuk membangun grafik **Loss vs. Epoch** dan **Accuracy vs. Epoch**
  sebagai bukti visual stabilitas dan konvergensi model selama proses pelatihan.

Tampilan antarmuka web klasifikasi awan
Gambar berikut menunjukkan tampilan antarmuka (*user interface*) aplikasi klasifikasi
citra awan berbasis Streamlit yang digunakan untuk proses inferensi dan visualisasi
hasil evaluasi model.

<img src="assets/Tampilan interface.jpeg" width="700">

Tampilan confusion matrix

<img src="assets/confusion matrix.jpeg" width="700">


## Integrasi Berkas Model dan Aplikasi Web

Berkas-berkas hasil pelatihan model digunakan secara langsung dalam aplikasi web
untuk mendukung proses inferensi dan penyajian hasil klasifikasi. Seluruh aset
tersebut dimanfaatkan tanpa memerlukan proses pelatihan ulang model.

### Fungsi Berkas dalam Aplikasi
Berkas-berkas tersebut digunakan dalam aplikasi web untuk memfasilitasi:

1. **Pemuatan model secara langsung** dari berkas hasil pelatihan tanpa proses
   *retraining*.
2. **Prediksi citra awan** berdasarkan gambar yang diunggah oleh pengguna
   (*image upload*).
3. **Penyajian hasil klasifikasi** dalam bentuk label kelas awan dan nilai
   probabilitas prediksi.

---

### Arsitektur Aplikasi Web
Aplikasi web dirancang menggunakan **Flask/FastAPI** sebagai *backend* untuk
mengelola logika inferensi dan pemanggilan model, sementara antarmuka pengguna
dikembangkan menggunakan **HTML, CSS, dan JavaScript**.

Proses inferensi model sepenuhnya dilakukan secara **server-side** untuk memastikan
hasil prediksi yang konsisten, aman, dan dapat diakses secara independen dari
spesifikasi perangkat pengguna.

---

### Pendekatan Deployment
Pendekatan *deployment* yang digunakan mengikuti pola yang telah diterapkan secara
berhasil dalam penelitian sebelumnya. Integrasi model CNN ke dalam aplikasi web
interaktif sejalan dengan pendekatan yang diusulkan oleh **Tirtana et al. (2021)**,
yang menunjukkan bahwa sinergi antara model *Deep Learning* dan aplikasi web
merupakan solusi efektif dalam konteks identifikasi objek visual.

## 4. Hasil dan Pembahasan

### 4.1 Gambaran Umum Proses Pelatihan Model

Model klasifikasi citra awan pada penelitian ini dilatih menggunakan arsitektur
**Convolutional Neural Network (CNN)–Residual Network (ResNet)**. Proses pelatihan
dilakukan selama **40 epoch** dengan memanfaatkan **optimizer AdamW**, **scheduler
OneCycleLR**, serta teknik **gradient clipping sebesar 0.15** untuk menjaga stabilitas
pelatihan dan mencegah eksploitasi gradien yang berlebihan.

Seluruh proses pelatihan menghasilkan beberapa artefak utama yang digunakan baik
untuk evaluasi maupun deployment model, yaitu:

1. **`cloud_resnet_state_dict.h5`**  
   Berisi bobot (*weights*) final model CNN hasil pelatihan.

2. **`training_history.json`**  
   Menyimpan rekaman nilai *loss* dan *accuracy* pada setiap epoch selama proses
   pelatihan.

3. **`cloud_metrics.json`**  
   Berisi hasil evaluasi model pada data validasi, meliputi nilai *precision*,
   *recall*, *F1-score*, dan *accuracy*.

4. **`cloud_labels.json`**  
   Menyimpan label kelas awan yang digunakan dalam proses inferensi dan deployment
   aplikasi web.

5. **Visualisasi evaluasi model**  
   Meliputi *confusion matrix* dan *classification report* yang digunakan untuk
   menganalisis performa klasifikasi secara lebih mendalam.

Keberadaan berkas-berkas tersebut menunjukkan bahwa model telah siap untuk
diintegrasikan dan digunakan pada **web interface** tanpa memerlukan proses
pelatihan ulang.

---

## Hasil Pelatihan (Training Results)

### Perkembangan Training Loss dan Validation Loss

Selama 40 epoch pelatihan, nilai **training loss** mengalami penurunan secara
konsisten. Hal ini menunjukkan bahwa model mampu mempelajari pola visual pada dataset
awan secara progresif. Sementara itu, **validation loss** cenderung stabil pada nilai
yang relatif rendah setelah beberapa epoch awal.

Kondisi ini mengindikasikan bahwa:

- Model tidak mengalami *overfitting* yang berlebihan
- Variasi data dapat dipelajari tanpa penurunan performa yang signifikan pada data
  validasi

Stabilitas proses pelatihan ini dipengaruhi oleh pemilihan *hyperparameter* yang
tepat, khususnya penggunaan **scheduler OneCycleLR** yang mengatur perubahan *learning
rate* secara dinamis. Pendekatan ini sejalan dengan prinsip pelatihan adaptif pada
model *Deep Learning* modern.

Proses pelatihan

<img src="assets/proses training1.jpeg" width="800">

<img src="assets/proses training2.jpeg" width="800">

Grafik training loss

<img src="assets/grafik training loss.jpeg" width="800">

<img src="assets/grafik accuracy.jpeg" width="800">


#### 4.2.2 Evaluasi Model

Setelah proses pelatihan model **CNN–ResNet** selesai, tahap selanjutnya adalah
melakukan evaluasi kinerja model menggunakan **dataset validasi**. Evaluasi ini
bertujuan untuk mengukur kemampuan model dalam mengenali dan membedakan jenis awan
berdasarkan pola visual yang dipelajari selama proses pelatihan.

Data hasil evaluasi disimpan dalam berkas **`cloud_metrics.json`**, yang mencakup
berbagai metrik evaluasi, antara lain:
- Akurasi (*accuracy*)
- Precision
- Recall
- F1-score
- Confusion matrix

Evaluasi dilakukan secara sistematis untuk memastikan bahwa model tidak hanya mampu
mengenali pola pada data pelatihan, tetapi juga memiliki kemampuan **generalisasi yang
baik** ketika dihadapkan pada data baru yang belum pernah dilihat sebelumnya.

Model yang dikembangkan menggunakan kombinasi arsitektur **CNN–ResNet**, optimizer
**AdamW**, scheduler **OneCycleLR**, serta teknik **augmentasi citra**. Secara teoritis,
kombinasi pendekatan ini memberikan potensi yang besar untuk mencapai performa
klasifikasi yang tinggi dan stabil pada berbagai kelas awan.

Confusion matrix

<img src="assets/confusion matrix2.jpeg" width="800">

#### 4.2.3 Analisis Confusion Matrix

Analisis *confusion matrix* dilakukan untuk mengidentifikasi pola prediksi model
terhadap kelas sebenarnya. Hasil analisis menunjukkan beberapa temuan penting terkait
stabilitas prediksi dan pola kesalahan klasifikasi.

**1. Kelas dengan Prediksi Sangat Stabil**  
Model menunjukkan tingkat prediksi yang sangat baik pada kelas awan konvektif, yaitu:
- **Cumulus Clouds** → 63 prediksi benar dari 64 sampel
- **Cumulonimbus Clouds** → 38 prediksi benar dari 40 sampel

Rendahnya tingkat kesalahan pada kelas-kelas tersebut menunjukkan bahwa tekstur awan
konvektif memiliki karakteristik visual yang cukup unik dan mudah dipelajari oleh
model CNN–ResNet.

---

**2. Kelas yang Paling Sering Tertukar**  
Beberapa pola kesalahan klasifikasi yang dominan meliputi:

a. **Stratiform Clouds → High Cumuliform Clouds**  
Sebanyak 10 sampel salah diklasifikasikan. Hal ini diduga disebabkan oleh kemiripan
pola visual awan yang relatif datar dan berlapis sehingga terjadi *feature overlap*.

b. **Clear Sky → Stratocumulus Clouds**  
Sebanyak 10 kesalahan prediksi terjadi karena:
- Stratocumulus dapat memiliki tekstur yang sangat halus
- Citra cerah sebagian dapat menyerupai kondisi mendung tipis

c. **Cirriform Clouds tertukar dengan Cumulonimbus dan High Cumuliform**  
Jumlah data yang terbatas (*support* rendah) menyebabkan model kesulitan mempelajari
fitur halus awan cirrus yang cenderung tipis dan transparan.

---

#### 4.2.4 Akurasi Model

Model mencapai **akurasi keseluruhan sebesar 89,30%**, yang menunjukkan bahwa hampir
sembilan dari sepuluh citra awan dapat diklasifikasikan dengan benar. Nilai akurasi ini
mengindikasikan bahwa model memiliki kemampuan belajar yang kuat terhadap pola visual
awan, meskipun klasifikasi awan memiliki tingkat kompleksitas tinggi akibat variasi
bentuk, resolusi citra, kondisi pencahayaan, dan kemiripan antar kelas.

Dalam konteks literatur klasifikasi citra meteorologi, akurasi di atas **85%** umumnya
dianggap tinggi. Oleh karena itu, capaian akurasi **89,30%** menunjukkan bahwa
arsitektur model dan strategi pelatihan yang digunakan telah sesuai untuk tugas
klasifikasi awan.

---

#### 4.2.5 Kinerja Per Kelas (Per-Class Metrics)

Tabel berikut menyajikan ringkasan performa klasifikasi untuk setiap kelas awan
berdasarkan nilai **precision**, **recall**, **F1-score**, dan **support**.

| Kelas Awan | Precision | Recall | F1-score | Support |
|-----------|----------|--------|---------|--------|
| High Cumuliform Clouds | 0.816 | 0.922 | 0.866 | 77 |
| Cumulus Clouds | 0.969 | 0.984 | 0.977 | 64 |
| Cirriform Clouds | 0.538 | 0.636 | 0.583 | 11 |
| Stratiform Clouds | 0.954 | 0.858 | 0.903 | 120 |
| Stratocumulus Clouds | 0.875 | 0.883 | 0.879 | 103 |
| Cumulonimbus Clouds | 0.905 | 0.950 | 0.927 | 40 |
| Clear Sky | 0.910 | 0.859 | 0.884 | 71 |

---

**Temuan Penting:**

1. **Kelas dengan performa terbaik**  
   - *Cumulus Clouds* (F1-score = 0.977)  
   - *Cumulonimbus Clouds* (F1-score = 0.927)  
   - *Stratiform Clouds* (F1-score = 0.903)  

   Hasil ini menunjukkan bahwa model memiliki kemampuan yang sangat baik dalam
   mengenali pola awan konvektif dan awan berlapis yang umum muncul dalam kondisi
   atmosfer stabil.

2. **Kelas dengan performa terendah**  
   - *Cirriform Clouds* (F1-score = 0.583)  

   Rendahnya performa pada kelas ini merupakan kondisi yang umum terjadi karena:
   - Tekstur awan cirrus sangat tipis  
   - Warna cenderung mirip dengan *clear sky*  
   - Pola visual tidak sekuat awan rendah dan awan konvektif  

   Selain itu, kelas ini memiliki jumlah sampel paling sedikit (*support* = 11),
   sehingga model belum mampu mempelajari pola secara optimal.

3. **Rata-rata Macro F1-score**  
   Nilai **macro average F1-score sebesar 0.8599** menunjukkan bahwa model memiliki
   performa yang cukup seimbang di seluruh kelas. Kedekatan nilai macro F1-score
   dengan nilai akurasi mengindikasikan bahwa model tidak hanya akurat secara
   keseluruhan, tetapi juga memiliki keseimbangan yang baik antara *precision* dan
   *recall* pada setiap kelas.

   Dengan demikian, model tidak menunjukkan bias yang signifikan terhadap kelas
   mayoritas dan mampu berkinerja dengan baik pada seluruh **7 kelas awan**, termasuk
   kelas minoritas.

### 4.4 Pembahasan

Model **CNN–ResNet** yang dikembangkan dalam penelitian ini menunjukkan performa
klasifikasi yang kuat, tercermin dari **akurasi keseluruhan sebesar 89,3%** serta
nilai **macro average F1-score sebesar 85,99%**. Kedekatan nilai macro average
F1-score dengan nilai akurasi mengindikasikan bahwa model tidak hanya akurat secara
keseluruhan, tetapi juga memiliki keseimbangan yang baik antara *precision* dan
*recall* pada seluruh kelas. Dengan demikian, model tidak menunjukkan bias yang
signifikan terhadap kelas mayoritas dan mampu berkinerja baik pada seluruh **7 kelas
awan**, termasuk kelas dengan jumlah sampel yang lebih sedikit (*minority classes*).

Capaian performa ini tidak terlepas dari efektivitas arsitektur **Residual Network
(ResNet)** yang digunakan. Keberadaan *residual block* terbukti mampu mencegah masalah
*vanishing gradient*, sehingga model tetap dapat mempelajari pola visual yang kompleks
meskipun jaringan bersifat relatif dalam. Mekanisme *shortcut connection* memungkinkan
aliran gradien yang lebih stabil selama proses pelatihan, menghasilkan sensitivitas
yang tinggi terhadap struktur awan konvektif yang memiliki fitur global yang tegas.
Temuan ini konsisten dengan teori yang menyatakan bahwa ResNet memiliki kapabilitas
unggul dalam mengekstraksi fitur tingkat tinggi pada objek dengan kompleksitas visual
yang tinggi, termasuk citra atmosfer.

Di sisi lain, hasil penelitian juga menunjukkan bahwa **distribusi dataset** memiliki
pengaruh yang signifikan terhadap performa model. Kelas awan minoritas seperti
*Cirriform Clouds* menunjukkan performa yang jauh lebih rendah dibandingkan kelas
lainnya, dengan nilai **F1-score sebesar 0,58**. Model mengalami kesulitan dalam
mengenali awan cirrus yang memiliki karakteristik visual sangat tipis, intensitas
cahaya rendah, serta kemiripan yang tinggi dengan kondisi *clear sky*. Hal ini
mengindikasikan bahwa **ketidakseimbangan kelas (class imbalance)** berdampak langsung
terhadap kemampuan model dalam mempelajari representasi fitur yang kaya pada kelas
minoritas. Secara akademik, fenomena ini sejalan dengan literatur yang menyatakan bahwa
kinerja *recall* pada CNN sangat sensitif terhadap ukuran dan distribusi dataset,
khususnya pada skenario klasifikasi multi-kelas berbasis citra.

Selain distribusi data, **teknik augmentasi citra** juga terbukti berperan penting
dalam meningkatkan performa model. Augmentasi berupa rotasi dan *flipping* mampu
memperkaya variasi data pelatihan, terutama pada jenis awan konvektif seperti
*Cumulonimbus* dan *Cumulus* yang memiliki orientasi dan struktur dinamis di atmosfer.
Pendekatan ini terbukti meningkatkan nilai *recall* serta membantu model menghasilkan
generalisasi yang lebih stabil pada dataset validasi. Temuan ini sejalan dengan
penelitian **Tilasefana & Putra (2023)** yang menegaskan bahwa augmentasi spasial dapat
meningkatkan performa CNN dalam klasifikasi citra cuaca dan atmosfer dengan tingkat
variabilitas tinggi.

Meskipun demikian, model masih menghadapi tantangan dalam mengklasifikasikan awan
dengan tekstur sangat tipis seperti cirrus. Dari perspektif meteorologi, awan cirrus
memiliki ciri fisik berupa lapisan serat tipis, berwarna pucat, dan sering bercampur
dengan kondisi cerah. Karakteristik ini menyulitkan algoritma CNN yang cenderung
mengandalkan perbedaan tekstur dan intensitas piksel. Oleh karena itu, klasifikasi
awan tipis memerlukan pendekatan lanjutan, seperti *fine-grained classification*,
penerapan *attention mechanism* (misalnya **SE-Block** atau **CBAM**), maupun
penambahan data sintetis berbasis model generatif (**GAN**) untuk meningkatkan
sensitivitas model terhadap fitur visual yang halus.

Secara keseluruhan, pembahasan ini menegaskan bahwa meskipun model CNN–ResNet telah
menunjukkan performa yang sangat baik pada sebagian besar kelas awan, masih terdapat
aspek penting yang perlu diperhatikan pada penelitian selanjutnya. Aspek tersebut
meliputi ketidakseimbangan data, kesulitan klasifikasi awan tipis, serta kebutuhan
augmentasi dan strategi pembelajaran lanjutan. Dengan penyempurnaan pada aspek-aspek
tersebut, model berpotensi mencapai performa yang lebih tinggi dan memberikan
kontribusi yang signifikan dalam pengembangan **sistem klasifikasi awan otomatis**
untuk aplikasi meteorologi modern.

#### Contoh Hasil Klasifikasi 1
Hasil inferensi menunjukkan bahwa model memiliki tingkat keyakinan yang sangat tinggi
dalam melakukan prediksi. Model berhasil mengidentifikasi formasi awan sebagai
**High Cumuliform Clouds** dengan tingkat probabilitas sebesar **98,83%**.

Selain kelas dengan probabilitas tertinggi, model juga mendistribusikan nilai
probabilitas ke kelas awan lainnya. Distribusi ini mencerminkan bagaimana model
memproses fitur visual yang bersifat ambigu dan memiliki kemiripan antar kelas.

<img src="assets/conoth awan1.jpeg" width="600">


#### Contoh Hasil Klasifikasi 2
Pada pengujian lainnya menggunakan citra awan yang berbeda, model kembali menunjukkan
performa prediksi yang konsisten. Formasi awan berhasil diklasifikasikan sebagai
**Stratocumulus Clouds** dengan tingkat probabilitas sebesar **96,39%**.

Seperti pada pengujian sebelumnya, model tetap memberikan distribusi probabilitas
pada kelas-kelas lain. Hal ini menunjukkan bahwa model tidak hanya menghasilkan
keputusan klasifikasi akhir, tetapi juga merepresentasikan tingkat ketidakpastian
pada setiap kelas berdasarkan karakteristik visual citra masukan.

<img src="assets/conoth awan2.jpeg" width="600">


---

Pengujian ini menegaskan bahwa model CNN–ResNet yang dikembangkan mampu melakukan
klasifikasi awan secara **akurat dan stabil** pada data baru, serta memberikan
informasi probabilistik yang berguna untuk interpretasi hasil prediksi dalam konteks
aplikasi meteorologi.


---

## 🌐 Web Interface

Aplikasi web dikembangkan menggunakan **Streamlit** untuk inferensi real-time.

---

## 📁 Struktur Repository

```
├── dataset/
├── models_web_h5/
├── Klasifikasi_awan_resnet_cnn_pytorch.py
├── Cloud_klasifikasi_interface.py
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
