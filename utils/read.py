import pymupdf
import cv2
import pandas as pd
import re

def read_pdf(root):
    """reads the pdf

    returns the images from pdf
    """
    doc = pymupdf.open(root)
    images = []
    for page_index in range(len(doc)):
        page = doc[page_index] # get the page
        image_list = page.get_images()
    
        for image_index, img in enumerate(image_list, start=1): # enumerate the image list
            xref = img[0] # get the XREF of the image
            pix = pymupdf.Pixmap(doc, xref) # create a Pixmap
    
            if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            images.append(pix)
    return images

def create_dataframe(
    InvoiceNumber, 
    Date, 
    Amount, 
    Vendor,
    Services
)->pd.DataFrame:
    """converts the texts extracted from the pdf into dataframe
    DataFrame:
    InvoiceNumber:
    Date: date of expense
    Amount: Spent amount
    Vendor: vendor name
    Services: list of the services used or the products bought
    """

    data = { "InvoiceNumber": InvoiceNumber,
            "Date": Date,
            "Amount": Amount,
            "Vendor": Vendor,
            "Services": Services
    }

    df = pd.DataFrame(data)
    return df

def read_images(root):
    """reads images from root folder 
    """
    img = cv2.imread(root)
    return img

def get_receipt_attr(receipt):
    """creates a dataframe for receipt
    parameters: 
        receipt: receipt texts

    returns attributes of receipt
    """
