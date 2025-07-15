# src/image_summarizer.py
import logging
import base64
import time
from tqdm import tqdm
import google.generativeai as genai
from google.api_core import exceptions # Import exceptions for API errors

logger = logging.getLogger(__name__)

# --- Generation Configuration for LLM Calls ---
GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# --- Helper function for retries (adapted for multimodal input) ---
def make_llm_call_with_retry_multimodal(model, content_parts, element_type, element_key, max_retries=5):
    """
    Attempts to make an LLM call with retries for rate limit errors, for multimodal input.
    """
    retries = 0
    while retries < max_retries:
        try:
            responses = model.generate_content(
                content_parts,
                stream=False
            )
            return responses.text
        except exceptions.ResourceExhausted as e: # Catch 429 errors specifically
            retries += 1
            wait_time = 2 ** retries # Exponential backoff
            if hasattr(e, 'retry_delay') and e.retry_delay:
                # Use the recommended retry delay from the API response if available
                wait_time = max(wait_time, e.retry_delay.seconds)

            logger.warning(
                f"Rate limit hit for {element_type} {element_key}. "
                f"Retrying in {wait_time} seconds (Attempt {retries}/{max_retries}). Error: {e}"
            )
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Unexpected error during {element_type} summarization for element {element_key}: {e}")
            # Return a clear error message in the summary list
            return f"Error summarizing {element_type} {element_key}."

    logger.error(f"Max retries ({max_retries}) exceeded for {element_type} {element_key}. Skipping.")
    return f"Failed to summarize {element_type} {element_key} after multiple retries."


def image_to_text_summarizer(caption_image_dict, model_name="gemini-2.0-flash"):
    """
    Summarizes images (with or without captions) into text using Gemini Flash.
    Args:
        caption_image_dict (dict): Dictionary of image_key -> base64_data + caption_text + page_number.
        model_name (str): The name of the Gemini model to use for summarization.
    Returns:
        list: A list of image summaries.
    """
    logger.info("Starting image summarization.")
    image_summary_list = []
    image_exception_counter = 0

    # Initialize the image summarization model
    model_image_summarize = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="You are a highly skilled image analysis and summarization specialist. Your task is to provide a detailed summary of what the image is about, leveraging its caption if present, or analyzing the image content if no caption is provided. The summary should start with 'This context is coming from image: '",
        generation_config=GENERATION_CONFIG,
        safety_settings=None,
    )

    for image_key, image_info in tqdm(caption_image_dict.items(), desc="Summarizing images", unit="image"):
        try:
            decoded_image_data = base64.b64decode(image_info["base64_data"])
            caption_text = image_info["caption_text"]
            page_number = image_info["page_number"]

            # Determine mime type based on common image formats or try to infer
            # For simplicity, assuming JPEG for now, but in a real app, you might infer from image_path or magic bytes
            # Unstructured often extracts images as PNG or JPEG.
            mime_type = "image/jpeg" # Default, adjust if your images are mostly PNG etc.
            if image_info.get("image_path") and image_info["image_path"].lower().endswith(".png"):
                mime_type = "image/png"
            elif image_info.get("image_path") and image_info["image_path"].lower().endswith(".gif"):
                mime_type = "image/gif" # Add other types if needed

            image_part = {'data': decoded_image_data, 'mime_type': mime_type}

            # Construct the prompt parts
            prompt_parts = [
                f"Image on page {page_number}. Caption: {caption_text if caption_text else 'No caption provided.'}\n",
                image_part,
                # The system instruction already covers the analysis steps, so the prompt can be simpler.
                "Please analyze the image and its caption (if any) and provide a detailed summary. Start the summary with 'This context is coming from image: '."
            ]

            summary = make_llm_call_with_retry_multimodal(
                model_image_summarize,
                prompt_parts,
                "image",
                f"page {page_number} ({image_key})" # Use image_key for unique identification in logs
            )

            if "Error summarizing image" in summary or "Failed to summarize image" in summary:
                image_exception_counter += 1
            image_summary_list.append(summary)
            logger.debug(f"Finished processing image on page {page_number}.")

        except Exception as e:
            image_exception_counter += 1
            logger.error(f"Unhandled error during image processing for image_key {image_key}: {e}")
            image_summary_list.append(f"Error processing image with key {image_key}.")

    logger.info(f"Image summarization complete. Total images processed: {len(caption_image_dict)}")
    logger.info(f"Image summarization errors/failures: {image_exception_counter}")

    return image_summary_list
