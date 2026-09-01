# AI Document Summarizer & API

An AI-powered document summarization tool and REST API that can summarize `.txt`, `.pdf`, `.docx`, and `.md` files using a fine-tuned T5 model.

## Features
- **One-Click Run**: Auto-installs venv, dependencies, NLTK datasets, and model fallback.
- **Web UI & REST API**: Modern dark glassmorphic UI + OpenAPI/Swagger documentation.
- **Cross-Origin (CORS) Ready**: Connect any external frontend, mobile app, or backend service.
- **Document Support**: TXT, PDF (with OCR fallback for scans), DOCX, and Markdown.
- **Context-Aware Section Summarization**: Intelligently chunks long documents and outputs section key points.
- **Streaming & Non-Streaming**: Real-time response streaming for UI and standard JSON endpoints for APIs.

---

## Quick Start (One-Click)

Just double-click **`run.bat`** (or run `./run.bat` in terminal).

It will automatically:
1. Detect Python
2. Create virtual environment `.venv`
3. Install dependencies from `requirements.txt`
4. Download NLTK data
5. Download model fallback (`t5-small`) if no local model is found
6. Launch the server and open your browser to **http://127.0.0.1:8000**

---

## About: Dataset, Model & Technical Architecture

### 📊 Dataset Used
- **Dataset**: [CNN / DailyMail (Version 3.0.0)](https://huggingface.co/datasets/cnn_dailymail)
- **Split Sizes**:
  - **Train**: ~287,113 document-summary pairs
  - **Validation**: 13,368 pairs
  - **Test**: 11,490 pairs
- **Nature of Data**: Real-world news stories written by journalists paired with bulleted multi-sentence editorial highlights acting as ground-truth abstractive summaries.

### 🧠 Model Architecture & Training Setup
- **Base Architecture**: Google T5 (**Text-to-Text Transfer Transformer** - `t5-small`), an encoder-decoder model leveraging relative position embeddings and self-attention.
- **Task Formatting**: Prefix prompt conditioning (`summarize: <context> <text>`).
- **Label Masking**: Target pad tokens are masked to `-100` so cross-entropy loss ignores padding.
- **Optimization**: AdamW optimizer with a fine-tuning learning rate of `3e-5`, linear warmup/decay, and FP16 mixed precision.
- **Generation & Decoding**: Beam Search with `num_beams=4`, `length_penalty=1.2`, `min_length=30`, `max_length=150`, and `no_repeat_ngram_size=3` to avoid repetitive loops.
- **Evaluation Metrics**: ROUGE-1 (unigram overlap), ROUGE-2 (bigram overlap), and ROUGE-L (longest common subsequence).

### 📑 Document Ingestion & Section Intelligence
- **Multi-Format Extraction**:
  - `.pdf`: PyMuPDF (`fitz`) extracting structural text blocks with font-size heading detection.
  - `.docx`: `python-docx` parsing paragraph styles, bullet points, and tables.
  - `.txt` / `.md`: Native UTF-8 ingestion.
  - **OCR Fallback**: Tesseract OCR + Poppler (`pdf2image`) for scanned PDFs.
- **Two-Level Context-Aware Chunking**:
  1. Detects section boundaries and headings.
  2. Extracts document-level context overview (first 1–2 sentences).
  3. Synthesizes individual sections with injected global context.
  4. Generates an overall synthesis summary + numbered section key points.

---

## API Usage & Documentation

When the server is running, interactive API documentation is available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI Schema**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### 1. Health Check
```bash
curl -X GET http://127.0.0.1:8000/api/health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "supported_formats": [".docx", ".pdf", ".txt", ".text", ".md"],
  "max_upload_mb": 25
}
```

### 2. Summarize Raw Text (JSON)
```bash
curl -X POST http://127.0.0.1:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long document text here...", "max_length": 150, "structured": true}'
```
**Response:**
```json
{
  "summary": "**Summary** (500 words → 3 sections)\n...",
  "word_count": 500,
  "sections_processed": 3,
  "summary_length": 340
}
```

### 3. Summarize File (Multipart Upload)
```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@document.pdf"
```

### 4. Python Integration Example
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/summarize",
    json={"text": "Article content to summarize...", "structured": True}
)
print(response.json()["summary"])
```

---

## Configuration & Environment Variables

You can customize app parameters in [`src/config.py`](file:///c:/Users/ak612/Downloads/MY%20PC/txt%20summarizer%20project/txt%20summarizer%20project/src/config.py) or by passing environment variables:

| Variable | Default | Description |
|---|---|---|
| `APP_HOST` | `0.0.0.0` | Host interface |
| `APP_PORT` | `8000` | Port number |
| `APP_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `APP_MODEL_PATH` | `models/final` | Path to trained model directory |
| `APP_MAX_UPLOAD_MB` | `25` | Maximum file upload size in MB |

---

## Project Structure
```
├── src/                       # Python backend package
│   ├── __init__.py
│   ├── config.py              # Centralized configuration & CORS
│   ├── server.py              # FastAPI server (API endpoints + UI routes)
│   ├── summarizer.py          # Summarization logic (T5 model)
│   └── extractor.py           # Text extraction (PDF, DOCX, TXT, OCR)
├── static/                    # Frontend UI assets
│   ├── index.html             # Clean HTML markup
│   ├── css/
│   │   └── style.css          # Glassmorphism dark styles
│   └── js/
│       └── app.js             # Client-side streaming logic
├── training/                  # Model training scripts
│   ├── colab_train.py         # Google Colab GPU training script
│   └── summarizer_train.py    # Local training script
├── tests/                     # Tests
│   ├── test_pipeline.py       # End-to-end pipeline tests
│   ├── test_academic.py       # Academic text chunking test
│   └── fixtures/
│       └── test_academic.txt  # Sample academic text fixture
├── models/                    # Fine-tuned model weights (models/final)
├── tools/                     # Poppler binary for PDF rendering
├── uploads/                   # Temporary upload storage (gitignored)
├── docs/                      # Documentation
├── samples/                   # Sample testing files
├── requirements.txt           # Python dependencies
├── run.bat                    # One-click auto-setup launcher
├── train.bat                  # Training launcher
└── README.md
```

## Training
To train or fine-tune a model:
- Use `training/colab_train.py` on Google Colab (free T4 GPU recommended).
- Or run `train.bat` locally.
- Save output weights into `models/final/`.
