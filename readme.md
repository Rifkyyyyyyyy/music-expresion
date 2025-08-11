# Emotion-Based Music Recommender

Aplikasi berbasis Flask untuk mendeteksi emosi manusia dari gambar wajah, lalu merekomendasikan lagu berdasarkan emosi tersebut. Sistem ini menggunakan model CNN (Convolutional Neural Network) yang dilatih dari citra wajah dalam skala abu-abu, serta CSV berisi daftar lagu untuk setiap emosi.


![Image](https://github.com/user-attachments/assets/735b3a82-1d53-4949-9d38-e41c4af19c5a)

https://github.com/user-attachments/assets/30ddf624-b8e5-4c2c-a61b-ea6c4638396d

---

## ✨ Fitur

- Deteksi wajah menggunakan Haar Cascade  
- Klasifikasi emosi menggunakan CNN  
- Rekomendasi lagu berdasarkan emosi  
- Endpoint API untuk input gambar dalam format base64  
- Mendukung pagination untuk hasil rekomendasi musik  
- Backend Flask yang bersih dan modular  

---

## 🗂️ Struktur Direktori

```
.
├── app.py
├── model/
│   ├── emotion_model.h5
│   └── model_weights.h5
├── songs/
│   ├── happy.csv
│   ├── sad.csv
│   └── ...
├── template/
│   └── index.html
├── haarcascade_frontalface_default.xml
└── README.md
```

---

## 📥 Dataset

Dataset yang digunakan untuk melatih model emosi adalah **FER2013**.  
Silakan unduh di sini:

👉 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Setelah diunduh dan diekstrak, struktur folder sebaiknya menjadi seperti ini:

```
data/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

---

## ⚙️ Kebutuhan

- Python 3.7+
- Flask
- TensorFlow
- OpenCV
- Pandas
- NumPy
- Pillow
- Matplotlib *(opsional, untuk visualisasi training)*

---

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Rifkyyyyyyyy/music-expresion.git
cd music-expresion

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Siapkan File

- Letakkan model hasil training di folder `model/`
- Letakkan file lagu di folder `songs/`
- Letakkan dataset FER2013 di folder `data/train/` dan `data/test/`

---

## ▶️ Menjalankan Aplikasi

```bash
python app.py
```

Buka browser dan kunjungi: [http://localhost:5000](http://localhost:5000)

---

## 📡 API Endpoint

### `POST /detect_emotion`

#### Request (JSON)

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

#### Query Parameters (opsional)

- `page` – Halaman saat ini (default: `1`)
- `limit` – Jumlah lagu per halaman (default: `10`)

#### Response (JSON)

```json
{
  "emotion": "Happy",
  "music": {
    "total": 42,
    "page": 1,
    "limit": 10,
    "data": [
      {
        "Name": "Lagu A",
        "Album": "Album A",
        "Artist": "Artis A"
      }
    ]
  }
}
```

---

## 👨‍💻 Author

**Rifky Firmansyah**  
Universitas Pamulang

---

> Feel free to customize, improve, or extend the functionality based on your dataset or use case.
