from app import app
from flask import request, jsonify, render_template
from PIL import Image
import pytesseract
import numpy as np
from app.model import proses_image, model_segmentation
import tempfile
import cv2
import re

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image = request.files['file']

    if image.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = Image.open(image)
    
    #img = model_segmentation(img)
    
    #img_pil = Image.fromarray(img)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_file_path = temp_file.name
        img.save(temp_file_path)

    img_processed = proses_image(temp_file_path)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        np.save(temp_file, img_processed)

    img_processed_loaded = np.load(temp_file.name)
    
#################################################################################################
        
    #Custom class for declare region of interest.
    class ImageConstantROI():
        class KTP(object):
            ROIS = {
                "nik": [(290, 120 , 874, 189)],
                "nama": [(305, 190, 890, 240)],
                "tempat_tanggal_lahir": [(305, 230, 740, 270)],
                "jenis_kelamin": [(300, 260, 571, 300)],
                "gol_darah": [(580, 261, 864, 300)],
                "alamat": [(35, 290, 790, 455)],
                "agama": [(302, 420, 520, 453)],
                "status_perkawinan": [(300, 445, 590, 485)],
                "pekerjaan": [(302, 480, 785, 520)],}

            CHECK_ROI = [(930, 608, 1145, 736)]

    def crop_image(image, roi_coordinates):
        roi_cropped = image[
            roi_coordinates[1]:roi_coordinates[3],
            roi_coordinates[0]:roi_coordinates[2]]
        return roi_cropped

    def adjust_roi_coordinates(roi, scale_x, scale_y):
        adjusted_roi = [
            int(roi[0] * scale_x),
            int(roi[1] * scale_y),
            int(roi[2] * scale_x),
            int(roi[3] * scale_y)]
        return adjusted_roi

    def crop_image_by_roi(image, roi_name, scale_x, scale_y):
        roi_coordinates = ImageConstantROI.KTP.ROIS[roi_name][0]
        adjusted_roi = adjust_roi_coordinates(roi_coordinates, scale_x, scale_y)
        cropped_image = crop_image(image, adjusted_roi)
        return cropped_image

    # Read Image
    base_width = 1200
    base_height = 750
    base_img = cv2.cvtColor(img_processed_loaded, cv2.COLOR_RGB2BGR)
    
    # Actual Image
    base_h, base_w, base_c = base_img.shape
    
    # Calculate Coordinat
    scale_x = base_w / base_width
    scale_y = base_h / base_height
    
    # Cropping by class
    NIK = crop_image_by_roi(base_img, "nik", scale_x, scale_y)
    Nama = crop_image_by_roi(base_img, "nama", scale_x, scale_y)
    TTL = crop_image_by_roi(base_img, "tempat_tanggal_lahir", scale_x, scale_y)
    Kelamin = crop_image_by_roi(base_img, "jenis_kelamin", scale_x, scale_y)
    Goldarah = crop_image_by_roi(base_img, "gol_darah", scale_x, scale_y)
    Alamat = crop_image_by_roi(base_img, "alamat", scale_x, scale_y)
    Agama = crop_image_by_roi(base_img, "agama", scale_x, scale_y)
    Pernikahan = crop_image_by_roi(base_img, "status_perkawinan", scale_x, scale_y)
    Pekerjaan = crop_image_by_roi(base_img, "pekerjaan", scale_x, scale_y)
    
    # NIK
    def preprocessing_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0, sigmaY=0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        clahed_img = clahe.apply(blur)
        _, thresholded_img = cv2.threshold(clahed_img, thresh=165, maxval=255, type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        
        return thresholded_img
    
    NO_NIK = pytesseract.image_to_string(NIK, config= r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 --dpi 300')
    NO_NIK =  NO_NIK.replace('\n', '').replace('\x0c', '')

    # NAMA
    Nama_ktp = pytesseract.image_to_string(Nama, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    Nama_ktp = ''.join(filter(str.isalnum, Nama_ktp)).upper().replace('\n', '').replace('\x0c', '')
    Nama_ktp = Nama_ktp.replace("S", "").strip().replace("EE", "").strip()
    Nama_ktp = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', Nama_ktp)
    
    # TTL
    tempat_lahir = ""  
    tgl_lahir_formatted = ""
    Kelahiran = pytesseract.image_to_string(TTL, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    Kelahiran = re.sub(r'[^A-Za-z0-9]+', '', Kelahiran).upper()
    # Mencari posisi angka pertama
    match = re.search(r'\d', Kelahiran)
    if match:
        # Ekstraksi tempat lahir
        tempat_lahir = Kelahiran[:match.start()]

        # Ekstraksi tanggal lahir
        tgl_lahir = Kelahiran[match.start():].strip()

        # Format tanggal lahir
        tgl_lahir_formatted = f"{tgl_lahir[:2]}-{tgl_lahir[2:4]}-{tgl_lahir[4:]}"
    
    # Jenis Kelamin
    Jenis = pytesseract.image_to_string(Kelamin, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    Jenis = re.sub(r'[^A-Za-z0-9]+', '', Jenis).upper().replace("U", "I")
    jenis_kelamin_formatted = None
    # Ubah format jenis kelamin
    if 'LAKI' in Jenis:
        jenis_kelamin_formatted = "LAKI-LAKI"
    elif 'PEREM' in Jenis:
        jenis_kelamin_formatted = "PEREMPUAN"

    # Gol Darah
    darah = pytesseract.image_to_string(Goldarah, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    darah = re.sub(r'[^A-Za-z0-9]+', '', darah).replace("i", "l").upper()
    
    # Cek dan format golongan darah
    if 'A' in darah[len("GOLDARAH"):]:
        gol_darah_formatted = "A"
    elif 'B' in darah[len("GOLDARAH"):]:
        gol_darah_formatted = "B"
    elif 'O' in darah[len("GOLDARAH"):]:
        gol_darah_formatted = "O"
    elif 'AB' in darah[len("GOLDARAH"):]:
        gol_darah_formatted = "AB"
    else:
        gol_darah_formatted = "-"
    
    # Membuat dictionary dengan data
    data_gol_darah = gol_darah_formatted
    
    # ALAMAT
    alamat_ktp = pytesseract.image_to_string(Alamat, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    alamat_ktp = re.sub(r'[\n\x0c]', ' ', alamat_ktp).replace('Agama ISLAM', '').replace('Aiamat', 'Alamat').replace('Kei', 'Kel').replace('RWw', 'RW').replace('Kecamatan', 'Kecamatan ')

    parts = alamat_ktp.split()

    # Inisialisasi variabel untuk menyimpan data
    alamat = ""
    rt_rw = ""
    kel_desa = ""
    kecamatan = ""

    # Variabel untuk menghapus kata "Agama" dan "Islam" dari alamat
    remove_words = ["Agama", "Islam"]

    # Melakukan iterasi pada setiap bagian teks
    for i, part in enumerate(parts):
        if part == "Alamat":
            # Bagian setelah "Alamat" adalah alamat lengkap
            alamat = " ".join(parts[i+1:i+2])
        elif part.isdigit() and len(part) == 6:
            # Bagian yang mengandung 6 digit adalah RT/RW
            rt_rw = f"{part[:3]} / {part[3:]}"
        elif part.startswith("KelDesa") and i < len(parts) - 1:
            # Bagian yang dimulai dengan "KelDesa" adalah nama kelurahan/desa
            kel_desa = parts[i+1]
        elif part == "Kecamatan" and i < len(parts) - 1:
            # Mencari indeks kata "Kecamatan"
            kecamatan_index = parts.index(part)
            if kecamatan_index + 1 < len(parts):
                # Jika ditemukan kata "Kecamatan" dan ada kata setelahnya, maka isilah variabel kecamatan
                kecamatan = parts[kecamatan_index + 1]
    
    # Agama
    agama_ktp = pytesseract.image_to_string(Agama, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    agama_ktp = agama_ktp.strip()
    
    # Status perkawinan
    perkawinan = pytesseract.image_to_string(Pernikahan, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    perkawinan = re.sub(r'[^A-Za-z0-9]+', '', perkawinan).strip()
    if perkawinan == "BELUMKAWIN":
        perkawinan = "BELUM KAWIN"
    elif perkawinan == "CERAIHIDUP":
        perkawinan = "CERAI HIDUP"
    elif perkawinan == "CERAIMATI":
        perkawinan = "CERAI MATI"
    
    data_kawin = perkawinan
    
    # Pekerjaan
    kerja = pytesseract.image_to_string(Pekerjaan, config= r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6 --dpi 300')
    kerja = re.sub(r'[^A-Za-z0-9]+', '', kerja).strip()
    
    # ktp_data
    ktp_data = {
        "NIK": NO_NIK,
        "Nama": Nama_ktp,
        "Tempat Lahir": tempat_lahir,
        "Tanggal Lahir": tgl_lahir_formatted,
        "Jenis Kelamin": jenis_kelamin_formatted,
        "Golongan Darah": data_gol_darah,
        "Alamat": alamat,
        "RT/RW": rt_rw,
        "Kel/Desa": kel_desa,
        "Kecamatan": kecamatan,
        "Agama": agama_ktp ,
        "Status Perkawinan": data_kawin,
        "Pekerjaan": kerja
    }
    
    # Kembalikan hasil OCR dalam format JSON
    return jsonify({'text': ktp_data}), 200


if __name__ == '__main__':
    app.run(debug=True)
