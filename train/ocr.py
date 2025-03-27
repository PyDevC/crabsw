import pytesseract
import cv2
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
  "readpdf", r"C:\Users\ombeh\Downloads\crabsw\utils\readpdf.py")    

readpdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readpdf)

def detect_text(pdf_path):
    img = readpdf.extract_images(pdf_path)
    return pytesseract.image_to_string(img)