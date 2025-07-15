# src/pdf_extractor.py
import os
import logging
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

def extract_pdf_elements(path, fname, poppler_bin_path):
    """
    Extract images, tables, and chunk text from a PDF file.

    Args:
        path (str): Directory path of the PDF.
        fname (str): File name of the PDF.
        poppler_bin_path (str): Path to the Poppler bin directory.

    Returns:
        list: A list of unstructured elements extracted from the PDF.
    """
    logger.info("Running extract_pdf_elements function.")

    # Diagnostic: Check if pdfinfo.exe exists
    pdfinfo_exe = os.path.join(poppler_bin_path, "pdfinfo.exe")
    logger.debug(f"Checking for pdfinfo.exe at: {pdfinfo_exe}")

    if not os.path.exists(pdfinfo_exe):
        logger.critical(f"pdfinfo.exe not found at: {pdfinfo_exe}. Please verify your Poppler installation.")
        raise FileNotFoundError(f"CRITICAL: pdfinfo.exe not found at: {pdfinfo_exe}. Please verify your Poppler installation.")
    else:
        logger.info(f"pdfinfo.exe found at: {pdfinfo_exe}")

    # Save original PATH
    original_path = os.environ.get("PATH", "")

    # Prepend Poppler to PATH
    os.environ["PATH"] = f"{poppler_bin_path};{original_path}"
    os.environ["POPPLER_PATH"] = poppler_bin_path
    print(f"Diagnostic: PATH updated. New PATH starts with: {os.environ['PATH'][:100]}...")

    # --- Partition PDF ---
    try:
        logger.info(f"Reading PDF for file: {os.path.join(path, fname)} ...")
        if path: # If PDF_DIRECTORY is provided
            full_file_path = os.path.join(path, fname)
        else: # If PDF_FILE_NAME is already the full path
            full_file_path = fname
        raw_pdf_elements = partition_pdf(
            filename=os.path.join(path, fname),
            strategy='hi_res',
            extract_image_block_types=['Table', 'Image', 'Figure'],
            sort_mode='xy_cut',
            infer_table_structure=True,
            extract_forms=False,
            max_characters=4000,
        )
    finally:
        # Restore original PATH
        os.environ["PATH"] = original_path
        logger.info("Diagnostic: PATH restored to original state.")
    return raw_pdf_elements
