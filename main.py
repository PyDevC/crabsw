import os
from utils import readpdf, billprocess
import train.ocr as ocr

root_folder = os.getcwd()

for pdfs in os.walk(root_folder+r"\test"):
    img = readpdf.extract_images(pdfs)
    text = ocr.detect_text(img)
    text = billprocess.parse_text_file(text)
    results = billprocess.process_bill_text_file(text)
    print("Bills with invalid recipient (does not include 'truesales farma'):")
    print(results["invalid_recipients"].to_string(index=False), "\n")
    
    print("Duplicate bills (by bill_number):")
    print(results["duplicate_bills"].to_string(index=False), "\n")
    
    print("Bills with date mismatches (bill_date vs. expense_entry_date):")
    print(results["mismatched_dates"].to_string(index=False), "\n")
    
    print("Bills not related to official expenses (not hotel, transport, or food):")
    print(results["non_official_bills"].to_string(index=False), "\n")
    
    if results["multi_bill_pdf"]:
        print("Note: The text file contains multiple bills (likely from a multi-bill PDF).")
    else:
        print("The text file contains a single bill.")