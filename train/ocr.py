import pytesseract
import cv2
import importlib.util
import numpy as np

def detect_text(pdf_path):
    img = cv2.imread(pdf_path)
    return pytesseract.image_to_string(img)