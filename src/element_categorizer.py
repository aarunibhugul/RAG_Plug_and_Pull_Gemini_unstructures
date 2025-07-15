# src/element_categorizer.py
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def categorize_elements(raw_pdf_elements, pages_to_be_extracted=100):
    """
    Categorize extracted elements from a PDF into tables and texts.

    Args:
        raw_pdf_elements (list): List of unstructured.documents.elements.
        pages_to_be_extracted (int): Number of pages to extract from the beginning.

    Returns:
        tuple: A tuple containing lists of texts and tables (HTML).
    """
    logger.info("Categorization Pipeline started.")
    logger.info(f"Starting categorization of {len(raw_pdf_elements)} elements, extracting up to {pages_to_be_extracted} pages.")

    tables_html = []
    texts = []
    processed_elements = 0
    skipped_elements = 0

    # Use tqdm for progress bar
    for i in tqdm(range(len(raw_pdf_elements)), desc="Categorizing elements"):
        element = raw_pdf_elements[i]

        # Filter elements by page number if specified
        if pages_to_be_extracted > 0 and hasattr(element.metadata, 'page_number') and element.metadata.page_number > pages_to_be_extracted:
            skipped_elements += 1
            continue

        processed_elements += 1

        # Categorizing text elements
        if element.category in ["CompositeElement", "NarrativeText", "ListItem", "UncategorizedText"]:
            texts.append(str(element))  # Append raw text content
            logger.debug(f"Appended text element from page {element.metadata.page_number}, category: {element.category}")

        # Categorizing table elements
        elif element.category == "Table":
            table_html = element.metadata.text_as_html
            prev_element_text = ""

            if i > 0:
                prev_element = raw_pdf_elements[i-1]
                if prev_element.category in ["CompositeElement", "NarrativeText", "ListItem", "UncategorizedText"]:
                    prev_element_text = str(prev_element)

            if prev_element_text and len(prev_element_text) < 200:
                table_with_summary_appended = f"{prev_element_text}\n\n{table_html}"
                logger.debug(f"Appended table with preceding summary from page {element.metadata.page_number}")
            else:
                table_with_summary_appended = table_html
                logger.debug(f"Appended table without summary from page {element.metadata.page_number}")

            tables_html.append(table_with_summary_appended)

    logger.info(f"Categorization completed. Processed elements: {processed_elements}, skipped elements: {skipped_elements}")
    logger.info(f"Total texts extracted: {len(texts)}, total tables extracted: {len(tables_html)}")

    return texts, tables_html
