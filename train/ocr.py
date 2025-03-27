import pytesseract
import cv2
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
  "readpdf", r"C:\Users\ombeh\Downloads\crabsw\utils\readpdf.py")    

readpdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readpdf)
img, num_img = readpdf.extract_images(r"C:\Users\ombeh\Downloads\crabsw\test\1.pdf")
print(pytesseract.image_to_string(img))