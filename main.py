from utils.flag_receipt import flag, save_flags
from data import getimages
import utils.ocr as ocr
from utils.read import get_receipt_attr, create_dataframe
from train.test_receipt_type import receipt_type


receipts, cols, shape = getimages.get_hugging("Ananthu01/7000_invoice_images_with_json") # gets images of receipts

for receipt in receipts:    
    flags = {"personal": None, "duplicate": None, "date mismatch": None}
    receipt_text = ocr.read_image(receipts) # reading receipts
    InvoiceNumber, Date, Amount, Vendor, Services = get_receipt_attr(receipt_text)
    receipt_df = create_dataframe(InvoiceNumber, Date, Amount, Vendor, Services)
    flags["personal"] = receipt_type(receipt_df.Services)
    print(flags)
    
    if 0 in flags.values():
        save_flags(receipt_df, flags)
