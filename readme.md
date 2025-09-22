# The Legal Simplifier AI

An AI-powered tool to make legal documents understandable and actionable for everyone.

## Features

- **Document Upload:** Upload contracts in PDF or DOCX format.
- **Plain-Language Summaries:** Get clear, non-legalese summaries of complex documents.
- **Semantic Q\&A:** Ask questions about your document and receive context-backed answers.
- **Risk \& Obligation Analysis:** Instantly identify and categorize risks, obligations, and key action items.
- **Smart Date Extraction \& Validation:** Extract, highlight, and analyze important contract dates (expiry, signature, deadlines).
- **Document Comparison:** Compare two documents and see a markdown-style redline of differences.
- **User-Friendly Interface:** Streamlit-based, no legal or coding knowledge needed.


## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legal-simplifier-ai.git
cd legal-simplifier-ai
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


### 3. Set Up Your Environment

Create a `.env` file in the project directory and add your Gemini API key:

```
GEMINI_API_KEY=your_google_gemini_api_key
```


### 4. Run the App

```bash
streamlit run frontend.py
```


### Run Using Render

```bash
https://legalai-fugs.onrender.com
```




## Sample File

A sample contract for testing (`contract document.pdf`) is included in this repository.
**To test**, simply upload this file after launching the app.

## Usage

1. **Upload your contract** (or use the included sample).
2. **View the summary** and extracted risk/obligation information.
3. **Try asking questions** in plain English about the contract.
4. **Compare two contract versions** using the "Compare Documents" feature.

## Tech Stack

- Python, Streamlit
- LangChain, Hugging Face Transformers, ChromaDB
- Google Gemini API
- pdfplumber, python-docx


## Notes

- Requires Python 3.10 or higher.
- Recommended for legal assessments, clarity for non-lawyers, and contract analysis education.
- Do not use with confidential or sensitive real-world legal documents for production without review.


## License

This project is licensed under the MIT License.

***

Feel free to edit the description and URLs according to your actual repo and deployment details!

