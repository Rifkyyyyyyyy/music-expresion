from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import base64
import os
from io import BytesIO
from PIL import Image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '../template')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, '../model/emotion_model.h5')
CASCADE_PATH = os.path.join(BASE_DIR, '../haarcascade_frontalface_default.xml')
SONGS_DIR = os.path.join(BASE_DIR, '../songs')

emotion_model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

music_dist = {}
for idx, emotion in emotion_dict.items():
    file_path = os.path.join(SONGS_DIR, f"{emotion.lower()}.csv")
    if os.path.exists(file_path):
        music_dist[idx] = file_path
    else:
        print(f"[WARNING] File CSV tidak ditemukan untuk emosi '{emotion}': {file_path}")

def music_rec(emotion_index, page=1, limit=10):
    path = music_dist.get(emotion_index)
    if not path or not os.path.exists(path):
        return {"total": 0, "page": page, "limit": limit, "data": []}
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        df = df[['Name', 'Album', 'Artist']].sort_values(by='Name')

        total = len(df)
        start = (page - 1) * limit
        end = start + limit
        paginated_df = df.iloc[start:end]

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "data": paginated_df.to_dict(orient='records')
        }
    except Exception as e:
        print(f"[ERROR] Gagal membaca CSV lagu: {e}")
        return {"total": 0, "page": page, "limit": limit, "data": []}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({"error": "No image data"}), 400

        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))

        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        result = {
            "emotion": "No face detected",
            "music": {"total": 0, "page": page, "limit": limit, "data": []}
        }

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            try:
                cropped = cv2.resize(roi_gray, (48, 48)).reshape(1, 48, 48, 1).astype("float32") / 255.0
                prediction = emotion_model.predict(cropped, verbose=0)
                maxindex = int(np.argmax(prediction))
                emotion = emotion_dict.get(maxindex, "Unknown")
                music = music_rec(maxindex, page=page, limit=limit)

                print(f"[INFO] Jumlah wajah terdeteksi: {len(faces)}")
                print(f"[INFO] Hasil prediksi: {emotion}")

                result["emotion"] = emotion
                result["music"] = music
            except Exception as e:
                print(f"[ERROR] Gagal prediksi emosi: {e}")
                result["emotion"] = "Prediction error"
            break

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        return jsonify({"error": "Internal server error"}), 500

def clean_csv_file(input_path):
    try:
        df = pd.read_csv(input_path, on_bad_lines='skip')
        df.to_csv(input_path, index=False, quoting=1)
        print(f"[CLEANED] {input_path}")
    except Exception as e:
        print(f"[ERROR] Cleaning CSV {input_path}: {e}")

def clean_all_song_csvs():
    for filename in os.listdir(SONGS_DIR):
        if filename.endswith('.csv'):
            clean_csv_file(os.path.join(SONGS_DIR, filename))

if __name__ == '__main__':
    clean_all_song_csvs()
    app.run(debug=True)
