import string
from lib.search_utils import (DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, CACHE_PATH, BM25_K1)
from nltk.stem import PorterStemmer
from collections import (defaultdict, Counter)
import pickle
import os
import sys
import math

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print(f"Index and docmap files not found in {CACHE_PATH}")
        sys.exit(1)

    
    query_tokens = tokenize_text(query)
    results = []
    doc_ids = set()
    for token in query_tokens:
        token_doc_ids = inverted_index.get_documents(token)
        for doc_id in token_doc_ids:
            if doc_id in doc_ids:
                continue
            doc_ids.add(doc_id)
            doc = inverted_index.docmap[doc_id]
            if not doc:
                continue
            results.append(doc)
            if len(results) >= limit:
                break
        
    return results
    

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stopwords = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stopwords:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print(f"Index and docmap files not found in {CACHE_PATH}")
        sys.exit(1)
    return inverted_index.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print(f"Index and docmap files not found in {CACHE_PATH}")
        sys.exit(1)
    return inverted_index.get_idf(term)

def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print(f"Index and docmap files not found in {CACHE_PATH}")
        sys.exit(1)
    return inverted_index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print(f"Index and docmap files not found in {CACHE_PATH}")
        sys.exit(1)
    return inverted_index.get_bm25_tf(doc_id, term, k1)

def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print(f"Index and docmap files not found in {CACHE_PATH}")
        sys.exit(1)
    return inverted_index.get_tf_idf(doc_id, term)



class InvertedIndex:
    
    def __init__(self):
        self.index = defaultdict(set) #term (str) -> list of doc_ids[int]
        self.docmap: dict[int, dict] = {} #doc_id (int) -> {id: int, title: str, description: str}
        self.term_frequncies = defaultdict(Counter) # doc_id (int) -> counter objects (term (str) -> frequency (int))
        
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequncies_path = os.path.join(CACHE_PATH, "term_frequncies.pkl")

    def build(self) -> None: #It should iterate over all the movies and add them to both the index and the docmap.
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self) -> None: #It should save the index and the docmap to a file.
        #create a folder called cache
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequncies_path, "wb") as f:
            pickle.dump(self.term_frequncies, f)
        print(f"Index and docmap saved to {CACHE_PATH}")

    def load(self) -> None:
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"Index and docmap files not found in {CACHE_PATH}")
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequncies_path, "rb") as f:
            self.term_frequncies = pickle.load(f)

    def get_documents(self, term: str) -> list[int]: #get the set of documents for a given token, and return them as a list, sorted in ascending order by document ID.
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term has multiple tokens")
        token = tokens[0]
        doc_ids = self.index.get(token, set())
        return sorted(list(doc_ids))
    
    def __add_document(self, doc_id: int, text: str) -> None: 
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequncies[doc_id].update(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term has multiple tokens")
        token = tokens[0]
        return self.term_frequncies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term has multiple tokens")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_count =len(self.index[token])
        return math.log((doc_count + 1) / (term_count + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term has multiple tokens")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_count = len(self.index[token])
        return math.log((doc_count - term_count + 0.5) / (term_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term:str, k1: float = BM25_K1) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term has multiple tokens")
        token = tokens[0]
        tf = self.get_tf(doc_id, token)
        bm25_tf_saturation = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf_saturation

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
        
    


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

            
