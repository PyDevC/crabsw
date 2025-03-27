import fitz
import cv2
import numpy as np

def extract_images(pdf_path):
    extracted_images = []

    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Iterate through pages of the PDF
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            images = page.get_images(full=True)

            # Extract images
            for _, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                image_bytes.save("image.png")

                # Convert the image bytes to a NumPy array for cv2
                image_np = np.frombuffer(image_bytes, np.uint8)
                image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                if image_cv2 is not None:
                    extracted_images.append(image_cv2)

        num_extracted_images = len(extracted_images)
    except Exception as e:
        print(f"An error occurred: {e}")

    return extracted_images, num_extracted_images
