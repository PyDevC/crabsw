import os
import PIL
import pdf2image
import fitz  # PyMuPDF
import numpy as np

def convert_pdf_to_images(
    pdf_path, 
    output_folder=None, 
    format='png', 
    dpi=300, 
    pages=None
):
    """
    Convert PDF to images with multiple library support
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str, optional): Folder to save images. 
                                       If None, uses PDF directory
        format (str, optional): Output image format (png, jpg, etc.)
        dpi (int, optional): Resolution of output images
        pages (list, optional): Specific pages to convert. 
                                If None, converts all pages
    
    Returns:
        list: List of image paths or image arrays
    """
    # Validate input
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.dirname(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert method 1: pdf2image (recommended)
    try:
        images = pdf2image.convert_from_path(
            pdf_path, 
            dpi=dpi, 
            fmt=format.lower(),
            output_folder=output_folder,
            pages=pages
        )
        
        # Save or return images
        image_paths = []
        for i, image in enumerate(images, 1):
            if pages and i not in pages:
                continue
            
            image_filename = os.path.join(
                output_folder, 
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i}.{format}"
            )
            image.save(image_filename)
            image_paths.append(image_filename)
        
        return image_paths
    
    except Exception as pdf2image_error:
        print(f"pdf2image method failed: {pdf2image_error}")
    
    # Convert method 2: PyMuPDF (alternative)
    try:
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Determine pages to convert
        if pages is None:
            pages = range(len(doc))
        
        for page_num in pages:
            page = doc[page_num]
            
            # Render page to an image
            pix = page.get_pixmap(dpi=dpi)
            
            # Convert to PIL Image
            img = PIL.Image.frombytes(
                "RGB", 
                [pix.width, pix.height], 
                pix.samples
            )
            
            # Save image
            image_filename = os.path.join(
                output_folder, 
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num+1}.{format}"
            )
            img.save(image_filename)
            image_paths.append(image_filename)
        
        doc.close()
        return image_paths
    
    except Exception as pymupdf_error:
        print(f"PyMuPDF method failed: {pymupdf_error}")
    
    # If both methods fail
    raise RuntimeError("Could not convert PDF to images. Check dependencies.")

# Example usage function
def demonstrate_pdf_conversion(sample_pdf_path):
    """Demonstrate conversion using a sample PDF"""
    
    try:
        # Convert entire PDF
        full_pdf_images = convert_pdf_to_images(
            sample_pdf_path, 
            output_folder="pdf_images", 
            dpi=300
        )
        print(f"Converted {len(full_pdf_images)} pages")
        
        # Convert specific pages
        specific_page_images = convert_pdf_to_images(
            sample_pdf_path, 
            pages=[1, 3],  # Convert first and third pages
            output_folder="specific_pages"
        )
        print(f"Converted {len(specific_page_images)} specific pages")
    
    except Exception as e:
        print(f"Conversion demonstration failed: {e}")