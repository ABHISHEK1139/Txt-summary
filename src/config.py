"""
Centralized configuration for the AI Document Summarizer.
Edit these values or override them with environment variables.
"""
import os


# ── Server ────────────────────────────────────────────────────
HOST = os.getenv("APP_HOST", "0.0.0.0")          # 0.0.0.0 = accessible on LAN
PORT = int(os.getenv("APP_PORT", "8000"))

# ── CORS (Cross-Origin Resource Sharing) ──────────────────────
# Allow any origin by default so external apps, mobile apps,
# React/Vue frontends, etc. can call the API freely.
# Restrict in production by setting APP_CORS_ORIGINS env var
# e.g. APP_CORS_ORIGINS=https://mysite.com,http://localhost:3000
CORS_ORIGINS = os.getenv("APP_CORS_ORIGINS", "*").split(",")

# ── Model ─────────────────────────────────────────────────────
MODEL_PATH = os.getenv("APP_MODEL_PATH", "models/final")
FALLBACK_MODEL = "t5-small"   # downloaded from HuggingFace if local missing

# ── Upload ────────────────────────────────────────────────────
UPLOAD_DIR = os.getenv("APP_UPLOAD_DIR", "uploads")
MAX_UPLOAD_MB = int(os.getenv("APP_MAX_UPLOAD_MB", "25"))

# ── Summarization ────────────────────────────────────────────
MAX_CHUNK_WORDS = 350         # words per section chunk
MAX_SUMMARY_LENGTH = 150      # max tokens in generated summary
MAX_SECTION_SUMMARY = 120     # max tokens per section summary

# ── File support ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".text", ".md"}
