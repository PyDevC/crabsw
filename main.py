from utils.flag_receipt import flag_personal
from data import getimages
import utils.ocr as ocr
from utils.read import get_receipt_attr, create_dataframe


receipts, cols, shape = getimages.get_hugging("Ananthu01/7000_invoice_images_with_json") # gets images of receipts

for receipt in receipts:    
    receipt_text = ocr.read_image(receipts) # reading receipts
    InvoiceNumber, Date, Amount, Vendor, Services = get_receipt_attr(receipt_text)
    receipt_df = create_dataframe(InvoiceNumber, Date, Amount, Vendor, Services)
