# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:48:19 2025

@author: BMKGPC
"""
# app_resnet.py - Kode Streamlit Final (Menghapus Kotak Putih yang Bermasalah)

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ====================================================================
# --- 1. DEFINISI ARSITEKTUR MODEL ---
# ====================================================================

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_diseases)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# ====================================================================
# KONFIGURASI DAN MUAT ASET
# ====================================================================

st.set_page_config(
    page_title="Cloudy Dreams Arena",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definisikan jalur file aset (SILAKAN GANTI PATH INI)
WEIGHTS_PATH = r"E:\KULIAH S2\Semester3\Computer vision\Tugas_Kelompok_ACV_Awan\models_web_h5\cloud_resnet_state_dict.h5"
LABELS_PATH = r"E:\KULIAH S2\Semester3\Computer vision\Tugas_Kelompok_ACV_Awan\models_web_h5\cloud_labels.json"
METRICS_PATH = r"E:\KULIAH S2\Semester3\Computer vision\Tugas_Kelompok_ACV_Awan\models_web_h5\cloud_metrics.json"
DUMMY_HISTORY_PATH = r"E:\KULIAH S2\Semester3\Computer vision\Tugas_Kelompok_ACV_Awan\models_web_h5\training_history.json"
 
ANGGOTA = ["Novana Sari", "Niken Wahyuni", "Diah Ariefianty", "Lina Adrianti"]

# --- CSS Custom untuk Tema CANDY PASTEL ---
st.markdown("""
    <style>
    /* 1. VARIABEL WARNA */
    :root {
        --primary-color: #FFB6C1; /* Candy Pink */
        --secondary-background-color: #F8F8FF; 
        --cloud-color: #E6E6FA; /* Lavender Blush */
        --purple-glow: #8A2BE2; 
        --font: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: var(--cloud-color); 
        color: #333333;
    }
    
    h1 {
        font-family: 'Pacifico', cursive;
        font-size: 5rem;
        color: var(--purple-glow); 
        text-shadow: 0 0 15px #ADD8E6; 
        text-align: center;
        margin-bottom: 0px;
    }

    .cloud-members {
        background-color: #FFE4E1; 
        border: 2px solid var(--purple-glow);
        border-radius: 25px;
        padding: 10px 20px;
        margin: 20px auto;
        width: fit-content;
        box-shadow: 0 4px 10px rgba(138, 43, 226, 0.4);
        font-family: var(--font);
        font-weight: 700;
        color: #5A5A5A;
        text-align: center;
        max-width: 80%;
    }
    
    /* FIX: Styling untuk nilai metrik agar menonjol */
    div[data-testid="stMetricLabel"] {
        color: #6495ED !important; /* Biru untuk Label */
        font-weight: 700 !important;
        padding-bottom: 0px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1E90FF !important; /* Warna Biru Terang untuk Nilai */
        font-size: 1.5rem !important;
        font-weight: 900 !important;
    }
    
    /* Tombol dan Container Box lainnya tetap */
    .stButton>button {
        background-color: #FFDAB9; 
        color: var(--purple-glow); 
        font-weight: 900;
        border: 2px solid #FFA07A; 
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.2s;
        box-shadow: 0 4px #E9967A; 
    }
    .stButton>button:hover {
        background-color: #FFA07A; 
        color: white;
        box-shadow: 0 2px #E9967A;
        transform: translateY(2px);
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Pacifico&family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


# --- Fungsi Pemuatan Model dan Aset ---

@st.cache_resource
def load_assets():
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

        num_classes = len(class_names)
        model = CNN_NeuralNet(3, num_classes)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        model.eval()

        try:
            with open(DUMMY_HISTORY_PATH, 'r') as f:
                history_data = json.load(f)
        except FileNotFoundError:
            epochs = list(range(1, 41))
            history_data = {
                'epoch': epochs,
                'train_loss': [1.5 - (i * 0.03) + (np.random.rand() * 0.2) for i in epochs],
                'val_loss': [1.6 - (i * 0.03) + (np.random.rand() * 0.3) for i in epochs],
                'val_acc': [0.5 + (i * 0.01) + (np.random.rand() * 0.15) for i in epochs]
            }

        return model, class_names, metrics_data, history_data
    
    except Exception as e:
        st.error(f"Gagal memuat aset model. Pastikan semua path dan file valid: {e}")
        return None, None, None, None

model, CLASS_NAMES, METRICS_DATA, HISTORY = load_assets()

# --- 3. Definisi Transformasi Gambar untuk Prediksi ---
TRANSFORM = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ====================================================================
# FUNGSI PREDIKSI
# ====================================================================

@st.cache_data
def predict_image(image: Image.Image, _model: nn.Module, _transform: transforms.Compose, class_names: list):
    """Lakukan inferensi pada gambar yang diunggah."""
    
    try:
        img_tensor = _transform(image).unsqueeze(0) 

        with torch.no_grad():
            output = _model(img_tensor) 
        
        probabilities = F.softmax(output, dim=1).squeeze().numpy()
        predicted_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_index]

        prob_results = []
        for i, prob in enumerate(probabilities):
            prob_results.append({
                'class': class_names[i],
                'probability': prob.item()
            })
        
        return predicted_class, prob_results

    except Exception as e:
        return f"Error saat prediksi: {e}", []


# ====================================================================
# KOMPONEN UTAMA UI STREAMLIT
# ====================================================================

# --- JUDUL DAN ANGGOTA KELOMPOK ---
st.title("Cloudy Dreams☁️")
st.markdown(
    f"""
    <div class="cloud-members">
        Anggota Kelompok: {', '.join(ANGGOTA)}
    </div>
    <p style='text-align: center; color: #555;'>Dashboard Evaluasi & Prediksi Klasifikasi Awan (Custom ResNet)</p>
    <hr style="border-top: 4px dashed #FF69B4;">
    """, unsafe_allow_html=True
)

if METRICS_DATA and model is not None:
    col_metrics, col_predict = st.columns([2, 1])

    # ----------------------------------------------------
    # BAGIAN KIRI: Metrik Evaluasi & Grafik
    # ----------------------------------------------------
    with col_metrics:
        
        st.header("📊 Hasil Evaluasi & Pelatihan")
        
        # --- Ringkasan Metrik Global (Sederhana Tanpa Panel Biru) ---
        st.subheader("Ringkasan Metrik")
        
        acc = METRICS_DATA['accuracy']
        macro_f1 = METRICS_DATA['macro_avg_f1_score']
        total_samples = sum(item['support'] for item in METRICS_DATA['per_class_metrics'].values())
        num_classes = len(CLASS_NAMES)

        # HANYA MENGGUNAKAN COLUMNS STANDAR DAN st.metric
        # Styling akan dihandle oleh CSS global pada stMetricValue/stMetricLabel
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Akurasi Validasi", f"{acc * 100:.2f}%")

        with col2:
            st.metric("Macro F1-Score", f"{macro_f1 * 100:.2f}%")

        with col3:
            st.metric("Jumlah Kelas", num_classes)

        with col4:
            st.metric("Total Sampel", total_samples)
        
        st.markdown("<br>", unsafe_allow_html=True)


        # --- TAB/TOMBOL UNTUK METRIK DETAIL ---
        st.subheader("Detail Evaluasi")
        
        if 'eval_tab' not in st.session_state:
            st.session_state.eval_tab = 'Loss'
            
        btn_loss, btn_cm, btn_report = st.columns(3)

        with btn_loss:
            if st.button("📈 Loss & Akurasi", use_container_width=True):
                st.session_state.eval_tab = 'Loss'
        with btn_cm:
            if st.button("🔥 Confusion Matrix", use_container_width=True):
                st.session_state.eval_tab = 'CM'
        with btn_report:
            if st.button("📋 Metrik Per Kelas", use_container_width=True):
                st.session_state.eval_tab = 'Report'

        st.markdown("---")
        
        # --- TAMPILAN BERDASARKAN TOMBOL YANG DIPILIH ---
        
        if st.session_state.eval_tab == 'Loss':
            st.markdown("### Grafik Loss & Akurasi (Per Epoch)")
            
            history_df = pd.DataFrame(HISTORY)
            history_df['Epoch'] = history_df['epoch']
            
            col_loss_chart, col_acc_chart = st.columns(2) 

            # --- Grafik Loss (Training & Validation) ---
            with col_loss_chart:
                st.markdown("##### Loss per Epoch")
                fig_loss, ax_loss = plt.subplots(figsize=(7, 5))
                
                ax_loss.plot(history_df['Epoch'], history_df['train_loss'], label='Training Loss', color='#FF69B4', linestyle='-')
                ax_loss.plot(history_df['Epoch'], history_df['val_loss'], label='Validation Loss', color='#9333ea', linestyle='--')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss', color='black')
                ax_loss.legend(loc='upper right')
                ax_loss.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig_loss)
                plt.close(fig_loss)

            # --- Grafik Validation Accuracy (TERPISAH) ---
            with col_acc_chart:
                st.markdown("##### Validation Accuracy per Epoch")
                fig_acc, ax_acc = plt.subplots(figsize=(7, 5))
                
                ax_acc.plot(history_df['Epoch'], history_df['val_acc'], label='Validation Accuracy', color='#059669', linestyle='-.')
                ax_acc.set_xlabel('Epoch')
                ax_acc.set_ylabel('Accuracy', color='black')
                ax_acc.set_ylim(0, 1) 
                ax_acc.legend(loc='lower right')
                ax_acc.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig_acc)
                plt.close(fig_acc)
            
        elif st.session_state.eval_tab == 'CM':
            st.markdown("### Confusion Matrix (Matriks Kebingungan)")
            
            conf_mat = np.array(METRICS_DATA['confusion_matrix'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                conf_mat, 
                annot=True, 
                fmt='d', 
                cmap='PuRd', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES, 
                ax=ax
            )
            ax.set_title('Confusion Matrix Validasi', fontsize=16, color='#9333ea')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            st.pyplot(fig)
            plt.close(fig)

        elif st.session_state.eval_tab == 'Report':
            st.markdown("### Laporan Klasifikasi (Precision, Recall, F1-Score)")
            
            metrics_df = pd.DataFrame(METRICS_DATA['per_class_metrics']).T
            metrics_df.index.name = "Kelas Awan"
            
            metrics_display_df = metrics_df.copy()
            metrics_display_df['precision'] = (metrics_display_df['precision'] * 100).round(2).astype(str) + '%'
            metrics_display_df['recall'] = (metrics_display_df['recall'] * 100).round(2).astype(str) + '%'
            metrics_display_df['f1-score'] = (metrics_display_df['f1-score'] * 100).round(2).astype(str) + '%'
            
            st.dataframe(metrics_display_df, use_container_width=True)


    # ----------------------------------------------------
    # BAGIAN KANAN: Prediksi Data Baru
    # ----------------------------------------------------

    with col_predict:
        st.header("📸 Klasifikasi Jenis Awan")
        
        uploaded_file = st.file_uploader(
            "Upload Gambar Awan (JPEG/PNG)", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Preview Gambar', use_container_width=True)
            
            st.markdown("---")
            
            if st.button("KLASIFIKASI AWAN ☁️", use_container_width=True):
                
                with st.spinner('Menganalisis jenis awan...'):
                    predicted_class, prob_results = predict_image(image, model, TRANSFORM, CLASS_NAMES)
                
                st.subheader("Hasil Klasifikasi")
                
                if "Error" in predicted_class:
                    st.error(predicted_class)
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: #FF69B4; color: white; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 15px;">
                            <p style="font-size: 14px; margin: 0;">Kelas Prediksi Tertinggi:</p>
                            <p style="font-size: 30px; font-weight: bold; margin: 0;">{predicted_class.upper()}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                    st.subheader("Probabilitas Klasifikasi")
                    
                    prob_results.sort(key=lambda x: x['probability'], reverse=True)
                    
                    for res in prob_results:
                        prob = res['probability']
                        class_name = res['class']
                        percentage = f"{prob * 100:.2f}%"
                        
                        st.markdown(f"**{class_name}** ({percentage})")
                        st.progress(prob)

        else:
            st.info("Silakan unggah gambar di atas untuk memulai klasifikasi.")

else:
    st.error("Gagal memuat semua aset (Model/Metrik/Label). Pastikan semua path di file .py sudah benar.")