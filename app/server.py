from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import shutil
import os
import traceback
import json
import logging
from extractor import get_text
from summarizer import (
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

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3:1b"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pre-load model at startup
try:
    load_local_model()
    logger.info("Model pre-loaded successfully")
except Exception as e:
    logger.warning(f"Model pre-load failed (will retry on first request): {e}")


@app.post("/chat")
async def chat(request: ChatRequest):
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


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = None
    try:
        # Validate file
        if not file.filename:
            return JSONResponse(status_code=400, content={"error": "No file provided"})
        
        ext = os.path.splitext(file.filename)[1].lower()
        supported = {'.pdf', '.docx', '.txt', '.text', '.md'}
        if ext not in supported:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {ext}. Supported: {', '.join(supported)}"}
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
            chunks = _split_into_sections(text, max_words=350)
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
            
            section_results = []  # list of (heading, summary)
            failed_sections = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    from summarizer import _detect_section_heading
                    heading = _detect_section_heading(chunk)
                    summary = _summarize_single_chunk(chunk, context=doc_overview, max_length=120)
                    if summary:
                        section_results.append((heading, summary))
                    
                    # Send progress update with heading so frontend can render live
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
        # Clean up uploaded file (optional — uncomment to auto-delete)
        # if path and os.path.exists(path):
        #     os.remove(path)
        pass

app.mount("/", StaticFiles(directory=".", html=True), name="static")
