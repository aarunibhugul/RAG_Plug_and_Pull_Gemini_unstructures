# src/main.py
import os
import logging
from dotenv import load_dotenv
from tqdm.contrib.logging import logging_redirect_tqdm
import google.generativeai as genai

# Import functions from your new modules
from src.pdf_extractor import extract_pdf_elements
from src.element_categorizer import categorize_elements
from src.image_processor import convert_images_to_base64, generate_caption_image_page_number
from src.text_table_summarizer import generate_table_text_summaries
from src.image_summarizer import image_to_text_summarizer # New import for image summarizer

# Import configurations from config.py
from src.config import PDF_DIRECTORY, PDF_FILE_NAME, POPPLER_BIN_PATH, MODEL_NAME, PAGES_TO_BE_EXTRACTED

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set in the environment.")
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the script.")
else:
    logger.info("GEMINI_API_KEY found in the environment.")
    # Configure genai globally here
    genai.configure(api_key=GEMINI_API_KEY)

# --- Unstructured Setup (OCR_AGENT) ---
os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"
logger.info("OCR_AGENT environment variable set.")


def run_pipeline():
    """
    Runs the complete PDF processing pipeline using configurations from config.py.
    """
    logger.info("--- Starting PDF processing pipeline ---")

    # Step 1: Extract PDF Elements
    logger.info("Step 1/5: Extracting raw elements from PDF...") # Updated step count
    try:
        raw_pdf_elements = extract_pdf_elements(PDF_DIRECTORY, PDF_FILE_NAME, POPPLER_BIN_PATH)
        logger.info(f"Step 1/5: Successfully extracted {len(raw_pdf_elements)} elements from the PDF.")
    except Exception as e:
        logger.exception(f"Step 1/5: Failed to extract PDF elements: {e}")
        return

    # Step 2: Categorize Elements (Text and Tables)
    logger.info("Step 2/5: Categorizing extracted elements into text and tables...") # Updated step count
    with logging_redirect_tqdm():
        texts, tables_html = categorize_elements(raw_pdf_elements, PAGES_TO_BE_EXTRACTED)
    logger.info(f"Step 2/5: Categorization complete. Found {len(texts)} text elements and {len(tables_html)} table elements.")

    # Step 3: Image Processing (Base64 Conversion & Caption Mapping)
    logger.info("Step 3/5: Starting image processing pipeline (Base64 conversion & caption mapping)...") # Updated step count
    logger.info("Step 3.1: Converting images to Base64 format...")
    with logging_redirect_tqdm():
        raw_image_list, image_data_dict = convert_images_to_base64(raw_pdf_elements)
    logger.info(f"Step 3.1: Converted {len(raw_image_list)} images to Base64.")

    logger.info("Step 3.2: Generating image-caption mappings and page numbers...")
    with logging_redirect_tqdm():
        caption_image_objects = generate_caption_image_page_number(
            raw_pdf_elements,
            image_data_dict
        )
    logger.info(f"Step 3.2: Generated {len(caption_image_objects)} image-caption objects.")
    logger.info("Step 3/5: Image processing complete.")

    # Step 4: Generate Text and Table Summaries
    logger.info("Step 4/5: Generating summaries for text and tables using LLM...") # Updated step count
    with logging_redirect_tqdm():
        text_summaries, table_summaries = generate_table_text_summaries(
            texts,
            tables_html,
            model_name=MODEL_NAME
        )
    logger.info(f"Step 4/5: Summarization complete. Generated {len(text_summaries)} text summaries and {len(table_summaries)} table summaries.")

    # Step 5: Generate Image Summaries
    logger.info("Step 5/5: Generating summaries for images using LLM...") # New step
    with logging_redirect_tqdm():
        image_summaries = image_to_text_summarizer(
            caption_image_objects,
            model_name=MODEL_NAME
        )
    logger.info(f"Step 5/5: Image summarization complete. Generated {len(image_summaries)} image summaries.")

    # --- Pipeline Summary ---
    print("\n--- PDF Processing Pipeline Summary ---")
    print(f"Total Text Elements Processed: {len(texts)}")
    print(f"Total Table HTML Elements Processed: {len(tables_html)}")
    print(f"Total Image-Caption Objects Generated: {len(caption_image_objects)}")
    print(f"Total Text Summaries Generated: {len(text_summaries)}")
    print(f"Total Table Summaries Generated: {len(table_summaries)}")
    print(f"Total Image Summaries Generated: {len(image_summaries)}") # New summary line
    print(f"AI Model Configured: {MODEL_NAME}")

    logger.info("--- PDF processing pipeline completed successfully ---")

    # Return all processed data
    return texts, tables_html, caption_image_objects, text_summaries, table_summaries, image_summaries

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        processed_data = run_pipeline()
    except Exception as e:
        logger.critical(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)
        print(f"\nPipeline failed due to an error: {e}")
