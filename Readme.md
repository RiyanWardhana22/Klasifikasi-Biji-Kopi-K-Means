# KLASIFIKASI MUTU BIJI KOPI MENGGUNAKAN MODEL RGB, GCLM, DAN K-MEANS

## Instalasi Library

pip install opencv-python numpy scikit-image scikit-learn pandas

## Alur Program

<ul>
      EKSTRAKSI FITUR   
      <li>
        ekstraksi_fitur_RGB: Mengekstrak fitur warna (mean dan standar deviasi)
        dari channel Red, Green, Blue
      </li>
      <li>
        ekstraksi_fitur_GCLM: Mengekstrak fitur tekstur menggunakan Gray-Level
        Co-occurrence Matrix (GLCM) termasuk contrast, correlation, energy, dan
        homogeneity
      </li>
    </ul>
    <ul>
      MEMUAT DATASET
      <li>Membaca semua gambar dari folder 'train'</li>
      <li>
        Untuk setiap gambar yang berhasil dibaca, mengekstrak fitur RGB dan GLCM
      </li>
      <li>Menggabungkan semua fitur dan menyimpan nama file gambar</li>
    </ul>
    <ul>
      MAIN FUNCTION
      <li>
        Memuat dataset: Memanggil load_dataset untuk mendapatkan fitur dan nama
        file
      </li>
      <li>
        Standarisasi fitur: Menggunakan StandardScaler untuk menormalisasi fitur
      </li>
      <li>
        Clustering dengan K-Means: Membagi data menjadi 2 cluster (Premium dan
        Cacat)
      </li>
      <li>
        Evaluasi model: Menghitung silhouette score untuk mengukur kualitas
        clustering
      </li>
      <li>Menyimpan hasil:</li>
      - Menyimpan hasil clustering ke file CSV - Menampilkan jumlah gambar per
      kategori - Menyimpan sampel gambar ke folder terpisah berdasarkan cluster
    </ul>
