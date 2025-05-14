import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import pandas as pd
import shutil

def ekstraksi_fitur_RGB(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_mean = np.mean(image_rgb, axis=(0, 1))
    rgb_std = np.std(image_rgb, axis=(0, 1))
    return np.concatenate([rgb_mean, rgb_std])

def ekstraksi_fitur_GCLM(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    return np.concatenate([contrast, correlation, energy, homogeneity])

def load_dataset(folder_train):
    features = []
    filenames = []
    
    for img_name in os.listdir(folder_train):
        img_path = os.path.join(folder_train, img_name)
        image = cv2.imread(img_path)
        
        if image is not None:
            rgb_features = ekstraksi_fitur_RGB(image)
            glcm_features = ekstraksi_fitur_GCLM(image)
            combined_features = np.concatenate([rgb_features, glcm_features])
            
            features.append(combined_features)
            filenames.append(img_name)
        else:
            print(f"Gagal memuat gambar: {img_path}")
    
    return np.array(features), filenames

def simpan_gambar_percluster(folder_train, results, output_dir, max_samples=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    nama_cluster = {0: 'Premium', 1: 'Cacat'}
    
    for cluster in [0, 1]:
        cluster_dir = os.path.join(output_dir, nama_cluster[cluster])
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        file_cluster = results[results['cluster'] == cluster]['filename'].head(max_samples)
        for fname in file_cluster:
            src_path = os.path.join(folder_train, fname)
            dst_path = os.path.join(cluster_dir, fname)
            shutil.copy(src_path, dst_path)

def main():
    folder_train = 'train' 
    output_dir = 'cnth_hasil_klasifikasi'  
    
    print("Memuat dataset...")
    X, filenames = load_dataset(folder_train)
    
    print("Menstandarisasi fitur...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Melakukan clustering dengan K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f} (lebih tinggi lebih baik, rentang -1 hingga 1)")
    
    results = pd.DataFrame({
        'filename': filenames,
        'cluster': cluster_labels,
        'category': ['Premium' if label == 0 else 'Cacat' for label in cluster_labels]
    })
    results.to_csv('hasil_klasifikasi.csv', index=False)
    print("Hasil clustering disimpan ke 'hasil_klasifikasi.csv'")
    
    cluster_counts = results['category'].value_counts()
    print("\nJumlah gambar per kategori:")
    print(f"Premium: {cluster_counts.get('Premium', 0)} gambar")
    print(f"Cacat: {cluster_counts.get('Cacat', 0)} gambar")
    
    print("Menyimpan sampel gambar ke folder 'cnth_hasil_klasifikasi/Premium' dan 'cnth_hasil_klasifikasi/Cacat'...")
    simpan_gambar_percluster(folder_train, results, output_dir)
    
    def classify_new_image(image_path, kmeans_model, scaler):
        image = cv2.imread(image_path)
        if image is None:
            return "Gagal memuat gambar"
        
        rgb_features = ekstraksi_fitur_RGB(image)
        glcm_features = ekstraksi_fitur_GCLM(image)
        combined_features = np.concatenate([rgb_features, glcm_features])
        
        combined_features_scaled = scaler.transform([combined_features])
        cluster = kmeans_model.predict(combined_features_scaled)[0]
        return 'Premium' if cluster == 0 else 'Cacat'

if __name__ == "__main__":
    main()