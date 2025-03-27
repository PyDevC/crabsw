import easyocr

reader = easyocr.Reader(['en']) 

def read_image(root):
    """returns text from the image
    """
    text = reader.readtext(root)
    return text
