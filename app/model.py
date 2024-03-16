import numpy as np
import cv2
import albumentations as albu
import torch

from PIL import Image, ExifTags
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from iglovikov_helper_functions.dl.pytorch.utils import rename_layers
from segmentation_models_pytorch import Unet
from torch import nn
from torch.utils import model_zoo
from midv500models.pre_trained_models import create_model

# iglovikov helver
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

# Model Segmentation
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import torch.nn.functional as F

'''
# load rgb
def load_rgb(image):
    print("Image path:", image)  # Mencetak path file gambar
    # Gunakan PIL untuk membuka gambar
    img = cv2.imread(image)

    # Periksa orientasi gambar
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        print("Exif data:", exif)  # Mencetak data Exif
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # lewati
        pass

    return img.convert('RGB')
'''

# model segmentasi
def model_segmentation(img):
    # Convert PIL Image to NumPy array
    image_np = np.array(img)

    # Tentukan path file model
    model = create_model("Unet_resnet34_2020-05-19")
    
    # Melakukan Transform
    transform = albu.Compose([albu.Normalize(p=1)], p=1)

    # Padded image
    padded_image, pads = pad(image_np, factor=32, border=cv2.BORDER_CONSTANT)

    # Padded image and tensor
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    # predict dari model
    with torch.no_grad():
        prediction = model(x)[0][0]

    # masking
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    # weighted image
    dst = cv2.addWeighted(image_np, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)

    # Mengaplikasikan mask ke gambar asli
    masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)

    # Menemukan kontur pada hasil masking
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Perform contour approximation
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Draw bounding box around contour
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(masked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menentukan ukuran frame matplotlib
    frame_width = 1200
    frame_height = 750

    # Membuat matriks transformasi perspektif dengan orientasi horizontal
    source_points = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])

    # Menentukan titik sudut gambar sasaran dengan orientasi horizontal (berdasarkan ukuran frame matplotlib)
    destination_points = np.float32([[0, 0], [frame_height, 0], [0, frame_width], [frame_height, frame_width]])

    # Membuat matriks transformasi perspektif
    perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

    # Melakukan transformasi perspektif pada gambar sumber hanya pada area mask
    straightened_image = cv2.warpPerspective(masked_image, perspective_matrix, (frame_height, frame_width))

    # Return the segmented image
    return straightened_image


def load_image(image_path):
    # Gunakan PIL untuk membuka gambar
    img_pil = Image.open(image_path)
    
    # Konversi gambar ke mode RGB
    img_pil = img_pil.convert('RGB')
    
    # Konversi gambar PIL menjadi array numpy
    img_np = np.array(img_pil)
    
    return img_np

# preprocessing image
def skew_correction(gray_image):
    orig = gray_image
    # threshold untuk menghilangkan noise tambahan
    thresh = threshold_otsu(gray_image)
    normalize = gray_image > thresh
    blur = gaussian(normalize, 3)
    edges = canny(blur)
    hough_lines = probabilistic_hough_line(edges)
    # hough lines mengembalikan daftar titik, dalam bentuk ((x1, y1), (x2, y2))
    # mewakili segmen garis. langkah pertama adalah menghitung kemiringan dari
    # garis-garis ini dari nilai titik yang dipasangkan
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]
    # kebetulan bahwa kemiringan ini juga adalah y di mana y = tan(theta), sudut
    # dalam sebuah lingkaran dengan garis yang di-offset
    rad_angles = [np.arctan(x) for x in slopes]
    # dan kita mengubah ke derajat untuk rotasi
    deg_angles = [np.degrees(x) for x in rad_angles]
    # nilai derajat
    histo = np.histogram(deg_angles, bins=100)
    # koreksi untuk penyelarasan 'ke samping'
    rotation_number = histo[1][np.argmax(histo[0])]
    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)
    # Rotasi gambar untuk menghilangkan kemiringan
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    # Sesuaikan sudut rotasi agar miring ke kiri
    rotation_number += 0.97  # Mengatur sudut rotasi
    # Mendapatkan matriks transformasi affine
    matrix = cv2.getRotationMatrix2D(center, rotation_number, 1.0)
    # Membuat gambar yang dirotasi
    rotated = cv2.warpAffine(orig, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated, rotation_number

def proses_image(img_path):
    # Baca gambar menggunakan OpenCV
    img = cv2.imread(img_path)

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize gambar
    #img_resized = cv2.resize(gray, (1200, 750))

    # Lakukan proses lainnya seperti yang Anda lakukan sebelumnya
    blur = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0, sigmaY=0)
    img, rotation_angle = skew_correction(blur)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, thresh=165, maxval=255, type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    img = cv2.copyMakeBorder(
        src=img,
        top=20,
        bottom=20,
        left=20,
        right=20,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255))
    img = cv2.resize(img, (1200, 750))

    return img