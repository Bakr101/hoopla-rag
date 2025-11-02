import json
import os
from typing import Any
from google import genai



GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
DEFAULT_ALPHA = 0.5
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH_MOVIES = os.path.join(PROJECT_ROOT, "data", "movies.json")
DATA_PATH_GOLDEN_DATASET = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
DATA_PATH_STOPWORDS = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
BM25_K1 = 1.5
BM25_B = 0.75

def load_movies() -> list[dict]:
    with open(DATA_PATH_MOVIES, 'r') as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(DATA_PATH_STOPWORDS, 'r') as f:
        return f.read().splitlines()

def load_golden_dataset() -> list[dict]:
    with open(DATA_PATH_GOLDEN_DATASET, 'r') as f:
        data = json.load(f)
    return data

def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "doc_id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def load_llm_client():
    return genai.Client(api_key=GEMINI_API_KEY)