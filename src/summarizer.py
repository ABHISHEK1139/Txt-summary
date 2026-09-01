from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import nltk
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Global model and tokenizer instances to avoid reloading
_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_local_model(model_path=None):
    global _model, _tokenizer
    if model_path is None:
        try:
            from src.config import MODEL_PATH, FALLBACK_MODEL
            model_path = MODEL_PATH
            fallback = FALLBACK_MODEL
        except ImportError:
            model_path = "models/final"
            fallback = "t5-small"
    else:
        fallback = "t5-small"

    if _model is None or _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            _model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
            _model.to(_device)
            _model.eval()
            print(f"Loaded local model from {model_path} on {_device}")
        except Exception as e:
            print(f"Error loading local model from {model_path}: {e}")
            print(f"Falling back to {fallback} from huggingface hub")
            _tokenizer = AutoTokenizer.from_pretrained(fallback)
            _model = AutoModelForSeq2SeqLM.from_pretrained(fallback)
            _model.to(_device)
            _model.eval()

def _is_heading(line):
    """Detect if a line is likely a heading/title, ignoring academic metadata."""
    line = line.strip()
    if not line:
        return False
        
    low_line = line.lower()
    
    # Ignore obvious academic metadata blocks (authors, affiliations)
    ignore_words = ['dr.', 'prof.', 'dept.', 'department', 'university', 'college', 'institute', 'email:', '@', 'received', 'abstract', 'keywords']
    if any(w in low_line for w in ignore_words) and len(line.split()) > 2:
        return False
        
    # Short lines that don't end with periods are likely headings
    if len(line.split()) < 10 and not line.endswith('.') and not line.endswith(','):
        if line.istitle() or line.isupper():
            return True
        if line[:2].replace('.', '').isdigit() or low_line.startswith(('chapter', 'section', 'part ')):
            return True
            
    # Lines that are ALL CAPS
    if line.isupper() and 2 <= len(line.split()) < 15:
        return True
        
    # Lines starting with numbers like "1." or "IV."
    if (line[:2].replace('.', '').isdigit() or low_line.startswith(('chapter', 'section', 'part '))) and len(line.split()) < 15:
        return True
        
    return False


def _split_into_sections(text, max_words=350):
    """Split text into context-aware chunks, keeping headings with their sections."""
    
    # Split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    # Group paragraphs into sections (heading + content)
    sections = []
    current_section = {"heading": "", "content": []}
    
    for para in paragraphs:
        lines = para.split('\n')
        first_line = lines[0].strip()
        low_first = first_line.lower()
        
        # STOP processing if we hit References or Bibliography
        if ('references' in low_first or 'bibliography' in low_first) and len(low_first.split()) < 5:
            break
            
        if _is_heading(first_line):
            # Save previous section
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"heading": first_line, "content": []}
            # Add remaining lines of this paragraph as content
            rest = '\n'.join(lines[1:]).strip()
            if rest:
                current_section["content"].append(rest)
        else:
            current_section["content"].append(para)
    
    if current_section["content"]:
        sections.append(current_section)
    
    # If no sections detected, treat entire text as one section
    if not sections:
        sections = [{"heading": "", "content": paragraphs}]
    
    # Build chunks from sections, respecting word limits
    chunks = []
    current_chunk_parts = []
    current_words = 0
    
    for section in sections:
        section_text = ""
        if section["heading"]:
            section_text = section["heading"] + "\n" + "\n\n".join(section["content"])
        else:
            section_text = "\n\n".join(section["content"])
        
        section_words = len(section_text.split())
        
        # If single section is too long, split by sentences
        if section_words > max_words:
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_words = 0
            
            # Keep heading with first sub-chunk
            sentences = sent_tokenize("\n\n".join(section["content"]))
            prefix = (section["heading"] + ": ") if section["heading"] else ""
            sent_chunk = [prefix] if prefix else []
            sent_words = len(prefix.split())
            
            for sent in sentences:
                sw = len(sent.split())
                if sent_words + sw > max_words and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_words = 0
                sent_chunk.append(sent)
                sent_words += sw
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))
        
        elif current_words + section_words > max_words and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = [section_text]
            current_words = section_words
        else:
            current_chunk_parts.append(section_text)
            current_words += section_words
    
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))
    
    return chunks if chunks else [text]


def _extract_doc_overview(text, max_words=50):
    """Extract a brief overview (first 1-2 sentences) for context."""
    sentences = sent_tokenize(text[:1000])  # Only look at first 1000 chars
    overview = ""
    for sent in sentences[:2]:
        if len(overview.split()) + len(sent.split()) <= max_words:
            overview += " " + sent
        else:
            break
    return overview.strip()


import re as _re

def _polish_text(text):
    """Post-process model output: capitalize, punctuate, clean up."""
    if not text:
        return text
    
    text = text.strip()
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Capitalize after periods
    def cap_after_period(match):
        return match.group(1) + match.group(2).upper()
    text = _re.sub(r'(\. )([a-z])', cap_after_period, text)
    
    # Ensure ends with period
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text


def _detect_section_heading(chunk_text):
    """Try to extract the heading from a chunk for structured output."""
    lines = chunk_text.strip().split('\n')
    for line in lines[:3]:
        line = line.strip()
        if line and _is_heading(line):
            return line
    return None


def _summarize_single_chunk(text, context="", max_length=150):
    """Summarize a single chunk with optional document context. Fail-safe."""
    if not text or not text.strip():
        return ""
    
    try:
        prefix = "summarize: "
        if context:
            input_text = prefix + f"Document about: {context}. Section: {text}"
        else:
            input_text = prefix + text

        inputs = _tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(_device)

        with torch.no_grad():
            summary_ids = _model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                num_beams=4,
                length_penalty=1.2,
                min_length=30,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        result = _tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return _polish_text(result)
    except Exception as e:
        logger.error(f"Error summarizing chunk: {e}")
        try:
            sentences = sent_tokenize(text)
            return _polish_text(" ".join(sentences[:2]))
        except:
            return text[:200] + "..."


def summarize_with_local_model(text, max_length=150, structured=True):
    """Summarize text of any length with context-aware chunk-and-merge. Fail-safe.
    
    If structured=True and document has multiple sections, returns formatted output
    with section headings and an overall summary.
    """
    if not text or not text.strip():
        return "Error: No text provided for summarization."
    
    try:
        load_local_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return f"Error: Could not load summarization model: {e}"

    try:
        chunks = _split_into_sections(text, max_words=350)
    except Exception as e:
        logger.error(f"Failed to split text: {e}")
        chunks = [text[:2000]]

    if len(chunks) == 1:
        return _summarize_single_chunk(chunks[0], max_length=max_length)

    # Extract overall document context
    doc_overview = _extract_doc_overview(text)
    
    logger.info(f"Long document: {len(chunks)} sections, context: {doc_overview[:80]}...")
    
    # Summarize each section and detect headings
    section_results = []  # list of (heading, summary)
    for i, chunk in enumerate(chunks):
        try:
            heading = _detect_section_heading(chunk)
            summary = _summarize_single_chunk(chunk, context=doc_overview, max_length=120)
            if summary:
                section_results.append((heading, summary))
            logger.info(f"  Section {i+1}/{len(chunks)} summarized")
        except Exception as e:
            logger.error(f"  Section {i+1}/{len(chunks)} failed: {e}")
            continue

    if not section_results:
        return "Error: Could not summarize any sections of the document."

    # Generate overall summary
    all_summaries = [s for _, s in section_results]
    merged = " ".join(all_summaries)
    overall = _summarize_single_chunk(merged, context=doc_overview, max_length=max_length)

    if not structured or len(section_results) <= 2:
        # For short docs, just return the overall summary
        return overall

    # Build clean structured output for long documents (3+ sections)
    output_parts = []
    
    # Overall summary as the main paragraph
    word_count = len(text.split())
    output_parts.append(f"**Summary** ({word_count} words → {len(section_results)} sections)\n")
    output_parts.append(overall)
    output_parts.append("\n\n---\n")
    output_parts.append("**Key Points by Section:**\n")
    
    # Merge consecutive sections without headings into one block
    for i, (heading, summary) in enumerate(section_results, 1):
        if heading:
            # Heading as bold prefix on same line as summary
            output_parts.append(f"• **{heading}:** {summary}")
        else:
            output_parts.append(f"• {summary}")
    
    return "\n".join(output_parts)


def stream_summarize_with_local_model(text, max_length=150):
    """Stream summarization for text of any length with context awareness."""
    load_local_model()

    chunks = _split_into_sections(text, max_words=350)

    if len(chunks) == 1:
        # Short text — stream directly
        prefix = "summarize: "
        input_text = prefix + text
        inputs = _tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(_device)
        streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            num_beams=1,
            streamer=streamer,
        )

        thread = Thread(target=_model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    # Long text — summarize sections with context, then stream the merge
    doc_overview = _extract_doc_overview(text)
    print(f"Long document: {len(chunks)} sections, summarizing with context...")
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = _summarize_single_chunk(chunk, context=doc_overview, max_length=120)
        chunk_summaries.append(summary)

    # Stream the final merge summary with document context
    merged = " ".join(chunk_summaries)
    prefix = "summarize: "
    input_text = prefix + f"Document about: {doc_overview}. Key points: {merged}"
    inputs = _tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(_device)
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_length=max_length,
        num_beams=1,
        streamer=streamer,
    )

    thread = Thread(target=_model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer

def summarize(text, ratio=0.07):
    sentences = sent_tokenize(text)

    if len(sentences) < 3:
        return text

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
    except ValueError:
        # Handle case where text might be empty or contain only stop words
        return text

    scores = X.sum(axis=1)

    # Calculate number of sentences to keep
    keep = max(1, int(len(sentences) * ratio))
    
    # Sort by score primarily
    # Note: X.sum returns a matrix, need to convert to array
    scores = np.asarray(scores).flatten()
    
    # Get indices of top `keep` sentences
    ranked_indices = np.argsort(scores)[::-1][:keep]
    
    # Sort indices to maintain original order in the summary
    ranked_indices = sorted(ranked_indices)

    selected = [sentences[i] for i in ranked_indices]

    return " ".join(selected)
