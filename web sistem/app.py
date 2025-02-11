from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from PIL import Image
import joblib
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import os

app = Flask(__name__)

# Load SVM model, scaler, dan label encoder
model = joblib.load('models/svm_model_linear2.pkl')
scaler = joblib.load('models/scaler2.pkl')
label_encoder = joblib.load('models/label_encoder2.pkl')

# Fungsi untuk menghitung GLCM dari gambar pada berbagai sudut
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    glcm = graycomatrix(img, distances=dists, angles=agls, levels=lvl, symmetric=sym, normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    feature.extend(glcm_props)
    feature.append(label)  # Tambahkan label di akhir
    return feature

# Fungsi untuk menghitung Local Binary Pattern (LBP) dari gambar
def extract_lbp_features(image, radius=1, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalisasi histogram
    return lbp_hist

# Fungsi untuk menghitung Sobel filter untuk deteksi tepi
def apply_sobel_filter(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return sobel.flatten()

# Fungsi untuk mengekstrak histogram intensitas dari gambar grayscale
def extract_intensity_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Fungsi untuk memproses gambar menjadi fitur GLCM, LBP, Sobel, dan histogram intensitas
def process_images_to_glcm_lbp_hist(data, labels):
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    combined_features = []

    for img, label in zip(data, labels):
        # Ekstraksi fitur GLCM
        glcm_features = calc_glcm_all_agls(img, label, props=properties)

        # Ekstraksi fitur LBP dengan radius lebih besar untuk acne nodules dan papula
        lbp_radius = 2 if label in ['acne nodules', 'papula'] else 1
        lbp_features = extract_lbp_features(img, radius=lbp_radius)

        # Ekstraksi Sobel untuk fungal acne
        sobel_features = apply_sobel_filter(img) if label == 'fungal acne' else np.zeros(128*128)

        # Ekstraksi histogram intensitas
        intensity_histogram = extract_intensity_histogram(img)

        # Gabungkan semua fitur menjadi satu vektor
        combined_features.append(np.hstack([glcm_features[:-1], lbp_features, sobel_features, intensity_histogram, label]))

    # Membuat nama kolom untuk dataframe
    columns = []
    angles = ['0', '45', '90', '135']
    for name in properties:
        for ang in angles:
            columns.append(name + "_" + ang)

    # Tambahkan nama kolom untuk fitur LBP, Sobel, dan histogram intensitas
    columns.extend([f'lbp_{i}' for i in range(len(lbp_features))])
    columns.extend([f'sobel_{i}' for i in range(128*128)])  # Sobel ukuran gambar
    columns.extend([f'intensity_hist_{i}' for i in range(256)])  # Histogram intensitas
    columns.append("label")

    # Mengonversi ke DataFrame
    combined_df = pd.DataFrame(combined_features, columns=columns)
    return combined_df

# Fungsi untuk memproses gambar input dan menghasilkan fitur untuk prediksi
def process_image(image):
    image = np.array(image.convert('L').resize((128, 128)))  # Konversi gambar ke grayscale dan resize
    labels = ['acne nodules']  # Dummy label untuk mengikuti struktur fungsi
    return process_images_to_glcm_lbp_hist([image], labels)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Proses file yang diunggah
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Konversi gambar untuk pemrosesan
    image = Image.open(file)
    
    # Ekstraksi fitur dari gambar
    df_features = process_image(image)
    
    # Pisahkan label dari fitur dan lakukan scaling
    features = df_features.drop('label', axis=1).values
    features = scaler.transform(features)
    
    # Pastikan fitur memiliki bentuk yang benar sebelum prediksi
    if features.shape[1] != 16674:  # Sesuaikan jumlah fitur sesuai model yang telah dilatih
        return jsonify({'error': f'Jumlah fitur tidak sesuai. Ditemukan {features.shape[1]} fitur, diharapkan 16674 fitur.'})
    
    # Prediksi menggunakan model SVM
    prediction = model.predict(features)[0]

    # Pastikan prediksi adalah integer
    print(f"DEBUG: Prediction value before conversion: {prediction}")
    prediction = int(prediction)  # Konversi prediksi menjadi integer
    print(f"DEBUG: Prediction value after conversion to integer: {prediction}")

    # Mengonversi prediksi menjadi label menggunakan label_encoder
    try:
        class_name = label_encoder.inverse_transform([prediction])[0]
    except IndexError as e:
        return jsonify({'error': f'IndexError: {str(e)} - Invalid index during inverse transform. Prediction: {prediction}'})
    
    # Contoh data penyebab dan solusi yang diperbarui
    solusi_data = {
        'Pustula': {
            'cause': 'Terbentuk karena pori-pori yang tersumbat oleh minyak (sebum) dan sel kulit mati, yang menyebabkan infeksi bakteri Propionibacterium acnes.',
            'solution': 'Membersihkan wajah secara teratur dengan pembersih yang mengandung asam salisilat atau benzoil peroksida.'
        },
        'papula': {
            'cause': 'Terbentuk dari pori-pori yang meradang akibat bakteri, minyak berlebih, dan penumpukan sel kulit mati.',
            'solution': 'Menggunakan produk perawatan yang mengandung bahan antiinflamasi seperti niacinamide.'
        },
        'acne fulminans': {
            'cause': 'Jenis jerawat yang sangat parah, biasanya terjadi secara tiba-tiba pada pria remaja atau dewasa muda.',
            'solution': 'Perawatan medis yang intensif, termasuk penggunaan antibiotik oral atau isotretinoin di bawah pengawasan dokter.'
        },
        'acne nodules': {
            'cause': 'Dihasilkan dari pori-pori yang tersumbat sangat dalam di bawah permukaan kulit, menyebabkan peradangan yang lebih serius.',
            'solution': 'Penggunaan obat oral seperti antibiotik atau isotretinoin, karena nodul sulit diobati hanya dengan obat topikal.'
        },
        'fungal acne': {
            'cause': 'Disebabkan oleh pertumbuhan berlebihan jamur Malassezia di folikel rambut.',
            'solution': 'Menggunakan pembersih atau sampo yang mengandung bahan antijamur seperti ketoconazole.'
        }
    }


    # Mengembalikan hasil prediksi dalam JSON
    result = {
        'penyakitNama': class_name,
        'penyebabDeskripsi': solusi_data.get(class_name, {}).get('cause', 'N/A'),
        'solusiDeskripsi': solusi_data.get(class_name, {}).get('solution', 'N/A')
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
