import easyocr

reader = easyocr.Reader(['en']) 

def read_image(root):
    """returns text from the image
    using easyocr
    parameter: root: path to image
    returns: extracted text
    """
    text = reader.readtext(root)
    return text
