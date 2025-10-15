import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH_MOVIES = os.path.join(PROJECT_ROOT, "data", "movies.json")
DATA_PATH_STOPWORDS = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

def load_movies() -> list[dict]:
    with open(DATA_PATH_MOVIES, 'r') as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(DATA_PATH_STOPWORDS, 'r') as f:
        return f.read().splitlines()