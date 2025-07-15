# src/config.py

# --- PDF Processing Configuration ---
# Directory where your PDF file is located
#PDF_DIRECTORY = "src"
PDF_DIRECTORY = "."

# Name of the PDF file to be processed
PDF_FILE_NAME = "Reliance_Annual_reportFY2021-22.pdf"

# Path to the 'bin' directory of your Poppler installation.
# Poppler is required by Unstructured for PDF processing.
# Example for Windows: r"C:\path\to\poppler-24.08.0\Library\bin"
# Example for Linux/macOS (often not needed if Poppler is in PATH, but can specify): "/usr/local/bin"
POPPLER_BIN_PATH = r"D:\popplers\poppler-24.08.0\Library\bin" # ✅ Update this to your actual Poppler path

# --- AI Model Configuration (for future use, if you integrate an LLM) ---
# Example: "gemini-1.5-pro", "gemini-1.0-pro", etc.
# This can be used later when you integrate LLM calls.
MODEL_NAME = "gemini-2.0-flash"

# --- Other Configurations ---
# Number of pages to extract from the beginning of the PDF.
# Set to 0 or a very large number to extract all pages.
PAGES_TO_BE_EXTRACTED = 100
