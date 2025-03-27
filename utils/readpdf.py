# import module
from pdf2image import convert_from_path
from PIL import Image

def extract_images(pdf_path):
    images = convert_from_path(pdf_path)
    image = Image(images)
