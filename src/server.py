from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import traceback
import json
import logging
from src.config import (
    CORS_ORIGINS, UPLOAD_DIR, MAX_UPLOAD_MB,
    SUPPORTED_EXTENSIONS, MAX_CHUNK_WORDS,
    MAX_SUMMARY_LENGTH, MAX_SECTION_SUMMARY,
)
from src.extractor import get_text
from src.summarizer import (
    summarize,
    summarize_with_local_model,
    stream_summarize_with_local_model,
    load_local_model,
    _split_into_sections,
    _extract_doc_overview,
    _summarize_single_chunk,
)

logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="AI Document Summarizer",
    description="Upload documents or paste text to get AI-powered summaries. "
                "Supports PDF, DOCX, TXT, and Markdown files.",
    version="1.0.0",
    docs_url="/docs",        # Swagger UI  → http://localhost:8000/docs
    redoc_url="/redoc",      # ReDoc       → http://localhost:8000/redoc
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────
# Allows any external app, frontend, or tool to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Pre-load model at startup ────────────────────────────────
try:
    load_local_model()
    logger.info("Model pre-loaded successfully")
except Exception as e:
    logger.warning(f"Model pre-load failed (will retry on first request): {e}")


# ══════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3:1b"

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = MAX_SUMMARY_LENGTH
    structured: bool = True

class SummarizeResponse(BaseModel):
    summary: str
    word_count: int
    sections_processed: int
    summary_length: int


# ══════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════

# ── Health / Info ─────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health():
    """Health check — returns server status and model availability."""
    from src.summarizer import _model
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "supported_formats": list(SUPPORTED_EXTENSIONS),
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/api/info", tags=["System"])
async def info():
    """Detailed metadata about dataset, model architecture, training, and pipeline."""
    from src.summarizer import _model, _device
    return {
        "project": "AI Document Summarizer",
        "dataset": {
            "name": "CNN / DailyMail",
            "version": "3.0.0",
            "train_samples": "~287,000 pairs",
            "validation_samples": "13,368 pairs",
            "test_samples": "11,490 samples",
            "description": "Real-world news stories with multi-sentence human summary highlights."
        },
        "model": {
            "base_model": "T5 (Text-to-Text Transfer Transformer)",
            "architecture": "Encoder-Decoder with Multi-Head Self-Attention",
            "device": str(_device),
            "loaded": _model is not None,
            "decoding": "Beam Search (k=4, length_penalty=1.2, no_repeat_ngram_size=3)",
            "optimizer": "AdamW (lr=3e-5) with FP16 Mixed-Precision",
            "evaluation_metrics": ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        },
        "pipeline": {
            "supported_formats": list(SUPPORTED_EXTENSIONS),
            "max_upload_size_mb": MAX_UPLOAD_MB,
            "max_chunk_words": MAX_CHUNK_WORDS,
            "ocr_support": "Tesseract OCR + Poppler fallback"
        }
    }


# ── Text Summarization (API) ─────────────────────────────────

@app.post("/api/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def api_summarize(request: SummarizeRequest):
    """
    Summarize raw text via the API.

    Send a JSON body with `text` and optionally `max_length` and `structured`.
    Returns a JSON response with the summary.

    **Example:**
    ```json
    POST /api/summarize
    {
        "text": "Your long document text here...",
        "max_length": 150,
        "structured": true
    }
    ```
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    try:
        load_local_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    try:
        summary = summarize_with_local_model(
            request.text,
            max_length=request.max_length,
            structured=request.structured
        )
        word_count = len(request.text.split())
        chunks = _split_into_sections(request.text, max_words=MAX_CHUNK_WORDS)
        return SummarizeResponse(
            summary=summary,
            word_count=word_count,
            sections_processed=len(chunks),
            summary_length=len(summary),
        )
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── File Upload Summarization (API) ──────────────────────────

@app.post("/api/upload", tags=["Summarization"])
async def api_upload(file: UploadFile = File(...)):
    """
    Upload a file and get its summary as JSON.

    Supports: PDF, DOCX, TXT, MD files (max 25MB).
    Returns the full summary in one response (no streaming).

    **Example:**
    ```
    curl -X POST http://localhost:8000/api/upload -F "file=@document.pdf"
    ```
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(path)
        if file_size > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large (max {MAX_UPLOAD_MB}MB)")

        text = get_text(path)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        load_local_model()
        summary = summarize_with_local_model(text)
        word_count = len(text.split())
        chunks = _split_into_sections(text, max_words=MAX_CHUNK_WORDS)

        return {
            "filename": file.filename,
            "summary": summary,
            "original_length": len(text),
            "word_count": word_count,
            "sections_processed": len(chunks),
            "summary_length": len(summary),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
#  FRONTEND ENDPOINTS (used by the web UI)
# ══════════════════════════════════════════════════════════════

@app.post("/chat", tags=["Frontend"])
async def chat(request: ChatRequest):
    """Chat endpoint used by the web UI. Streams text responses."""
    if not request.message or not request.message.strip():
        return JSONResponse(status_code=400, content={"error": "Empty message"})

    try:
        streamer = stream_summarize_with_local_model(request.message)

        def iter_content():
            try:
                for new_text in streamer:
                    yield new_text
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"\n[Error during summarization: {str(e)}]"

        return StreamingResponse(iter_content(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Chat error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload", tags=["Frontend"])
async def upload(file: UploadFile = File(...)):
    """File upload endpoint used by the web UI. Streams progress for long docs."""
    path = None
    try:
        if not file.filename:
            return JSONResponse(status_code=400, content={"error": "No file provided"})

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"}
            )

        # Save file
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(path)
        logger.info(f"📄 File saved: {file.filename} ({file_size/1024:.1f} KB)")

        # Extract text
        logger.info("📝 Extracting text...")
        text = get_text(path)

        if not text or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from file. The file may be empty, corrupted, or a scanned image without OCR support."}
            )

        word_count = len(text.split())
        logger.info(f"📊 Extracted {word_count} words from {file.filename}")

        # Ensure model is loaded
        try:
            load_local_model()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Model loading failed: {e}"})

        # Split into sections
        try:
            chunks = _split_into_sections(text, max_words=MAX_CHUNK_WORDS)
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            chunks = [text[:2000]]

        if len(chunks) == 1:
            # Short doc — return summary directly
            logger.info("Short document, single-pass summarization")
            summary = summarize_with_local_model(text)
            return {
                "summary": summary,
                "original_length": len(text),
                "word_count": word_count,
                "sections_processed": 1,
                "summary_length": len(summary)
            }

        # Long doc — stream section progress
        doc_overview = _extract_doc_overview(text)
        logger.info(f"Long document: {len(chunks)} sections, streaming progress")

        def process_sections():
            # Send metadata first
            yield json.dumps({
                "type": "info",
                "message": f"Processing {len(chunks)} sections ({word_count} words)...",
                "total_sections": len(chunks),
                "word_count": word_count
            }) + "\n"

            section_results = []
            failed_sections = 0

            for i, chunk in enumerate(chunks):
                try:
                    from src.summarizer import _detect_section_heading
                    heading = _detect_section_heading(chunk)
                    summary = _summarize_single_chunk(chunk, context=doc_overview, max_length=MAX_SECTION_SUMMARY)
                    if summary:
                        section_results.append((heading, summary))

                    yield json.dumps({
                        "type": "progress",
                        "section": i + 1,
                        "total": len(chunks),
                        "section_heading": heading or "",
                        "section_number": len(section_results),
                        "section_summary": summary or "(section skipped)"
                    }) + "\n"
                except Exception as e:
                    failed_sections += 1
                    logger.error(f"Section {i+1} failed: {e}")
                    yield json.dumps({
                        "type": "progress",
                        "section": i + 1,
                        "total": len(chunks),
                        "section_heading": "",
                        "section_number": len(section_results),
                        "section_summary": f"(failed: {str(e)[:50]})"
                    }) + "\n"

            if not section_results:
                yield json.dumps({
                    "type": "error",
                    "message": "Could not summarize any sections"
                }) + "\n"
                return

            # Generate overall summary from section summaries
            try:
                all_summaries = [s for _, s in section_results]
                merged = " ".join(all_summaries)
                overall = _summarize_single_chunk(merged, context=doc_overview, max_length=200)
            except Exception as e:
                logger.error(f"Merge failed: {e}")
                overall = " ".join([s for _, s in section_results[:3]])

            # Build structured markdown output
            parts = []
            parts.append(f"**📄 Summary** ({word_count:,} words → {len(section_results)} sections)\n")
            parts.append(overall)
            parts.append("\n\n---\n")
            parts.append("**📋 Key Points by Section:**\n")

            for i, (heading, summary) in enumerate(section_results, 1):
                if heading:
                    parts.append(f"**{i}. {heading}:** {summary}\n")
                else:
                    parts.append(f"**{i}.** {summary}\n")

            structured_summary = "\n".join(parts)

            yield json.dumps({
                "type": "complete",
                "summary": structured_summary,
                "original_length": len(text),
                "word_count": word_count,
                "sections_processed": len(section_results),
                "sections_failed": failed_sections,
                "summary_length": len(structured_summary)
            }) + "\n"

        return StreamingResponse(process_sections(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Upload error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        pass


# ── Static Files (must be last) ──────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
