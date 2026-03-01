# AI Document Summarizer

An AI-powered document summarization tool that can summarize `.txt`, `.pdf`, and `.docx` files using a fine-tuned T5 model.

## Features
- Upload and summarize documents (TXT, PDF, DOCX)
- Context-aware chunk-and-merge summarization for long documents
- Structured output with section headings
- Real-time streaming response
- Chat interface for asking questions
- Dark-themed modern UI

## Setup

### 1. Install Python dependencies
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install fastapi uvicorn python-docx nltk scikit-learn numpy torch transformers pymupdf pytesseract pdf2image pydantic
```

### 2. Download the model
The fine-tuned model is not included in this repo (too large for GitHub).

**Option A** — Place your trained model in `models/final/`:
```
models/
  final/
    config.json
    generation_config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
```

**Option B** — The app will automatically fall back to downloading `t5-small` from HuggingFace if no local model is found.

### 3. (Optional) Install PDF tools
For PDF extraction with OCR support, install:
- [Poppler](https://github.com/ossamamehmood/Poppler-Windows/releases) — add to `tools/` or system PATH
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) — install and add to PATH

### 4. Run the app
```bash
# Using the batch file:
run.bat

# Or manually:
cd app
python -m uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## Project Structure
```
├── app/
│   ├── server.py          # FastAPI backend
│   ├── summarizer.py      # Summarization logic (T5 model)
│   ├── extractor.py       # Text extraction (PDF, DOCX, TXT)
│   └── index.html         # Frontend UI
├── training/
│   ├── colab_train.py     # Google Colab training script
│   └── summarizer_train.py# Local training script
├── run.bat                # Quick-start script
├── train.bat              # Training launcher
└── sample.txt             # Sample file for testing
```

## Training
To train your own model, see `training/colab_train.py` (for Google Colab) or `training/summarizer_train.py` (for local training).
