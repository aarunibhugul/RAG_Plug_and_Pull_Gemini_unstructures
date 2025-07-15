# src/text_table_summarizer.py
import logging
from tqdm import tqdm
import google.generativeai as genai

logger = logging.getLogger(__name__)

# --- Generation Configuration for LLM Calls ---
GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# Removed api_key from function signature as it's now configured globally
def generate_table_text_summaries(texts, tables, model_name="gemini-2.0-flash"):
    """
    Generates summaries of text and tables using Gemini models.

    Args:
        texts (list): List of text elements (strings).
        tables (list): List of table HTML strings.
        model_name (str): The name of the Gemini model to use for summarization.

    Returns:
        tuple: Lists of text summaries and table summaries.
    """
    logger.info("Starting text and table summarization.")

    # Initialize Gemini models. api_key is no longer needed here as it's configured globally in main.py
    model_text_summarize = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="You are a financial genius tasked with summarizing text for retrieval. These summaries will be embedded and used to retrieve the raw text. Give a summary of the text covering the key aspects from the context ignoring any irrelevant information.",
        generation_config=GENERATION_CONFIG,
        safety_settings=None,
        # api_key=api_key # Removed this line, as it's handled by genai.configure()
    )

    model_table_summarize = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="You are a financial genius tasked with summarizing tables for retrieval. The summary should specify what the table represents and not give details on the numbers. Tables are represented in Markdown text format. The first line is present in the table. Review and utilize the line items if present in the table while generating the summary.",
        generation_config=GENERATION_CONFIG,
        safety_settings=None,
        # api_key=api_key # Removed this line, as it's handled by genai.configure()
    )

    text_summaries = []
    table_summaries = []

    total_elements_to_process = len(texts) + len(tables)
    processed_count = 0
    text_exception_counter = 0
    table_exception_counter = 0

    logger.info(f"Processing {len(texts)} text elements for summarization...")
    # Generate summaries of text
    for i, text in enumerate(tqdm(texts, desc="Summarizing texts", unit="text")):
        try:
            responses = model_text_summarize.generate_content(
                [text],
                stream=False
            )
            summary = responses.text
            text_summaries.append(summary)
            logger.debug(f"Successfully summarized text {i+1}/{len(texts)}")
        except Exception as e:
            text_exception_counter += 1
            logger.error(f"Text summarization error for text {i+1} ({text_exception_counter}): {e}")
            logger.debug(f"Problematic text (first 200 chars): {text[:200]}...")
            text_summaries.append(f"Error summarizing text: {text[:100]}...")
        processed_count += 1

    logger.info(f"Processing {len(tables)} table elements for summarization...")
    # Generate summaries for tables
    for i, table in enumerate(tqdm(tables, desc="Summarizing tables", unit="table")):
        try:
            responses = model_table_summarize.generate_content(
                [table],
                stream=False
            )
            summary = responses.text
            table_summaries.append(summary)
            logger.debug(f"Successfully summarized table {i+1}/{len(tables)}")
        except Exception as e:
            table_exception_counter += 1
            logger.error(f"Table summarization error for table {i+1} ({table_exception_counter}): {e}")
            logger.debug(f"Problematic table (first 200 chars): {table[:200]}...")
            table_summaries.append(f"Error summarizing table: {table[:100]}...")
        processed_count += 1

    logger.info(f"Summarization complete. Total elements processed: {processed_count}")
    logger.info(f"Text summarization errors: {text_exception_counter}")
    logger.info(f"Table summarization errors: {table_exception_counter}")

    return text_summaries, table_summaries
