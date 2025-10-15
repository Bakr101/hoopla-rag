import string
from .search_utils import (DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, CACHE_PATH)
from nltk.stem import PorterStemmer
from collections import (defaultdict, Counter)
from lib.search_utils import (load_movies, CACHE_PATH)
import pickle
import os
import sys

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
    
def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print(f"Index and docmap files not found in {CACHE_PATH}")
        sys.exit(1)
    return inverted_index.get_tf(doc_id, term)

def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    punctuation_translation = str.maketrans('', '', string.punctuation)
    text = text.translate(punctuation_translation)
    words = text.split(' ')
    stopwords = load_stopwords()
    for word in words:
        if word == ' ':
            words.remove(word)
        if word in stopwords:
            words.remove(word)
    return words

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def tokenize_text(text: str) -> list[str]:
    tokens = preprocess_text(text)
    stemmer = PorterStemmer()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stemmed_tokens = []
    for token in valid_tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens



class InvertedIndex:
    
    def __init__(self):
        self.index = defaultdict(set) #term (str) -> list of doc_ids[int]
        self.docmap = {} #doc_id (int) -> {id: int, title: str, description: str}
        self.term_frequncies = {} # doc_id (int) -> counter objects (term (str) -> frequency (int))
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequncies_path = os.path.join(CACHE_PATH, "term_frequncies.pkl")
    def __add_document(self, doc_id: int, text: str) -> None: 
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequncies[doc_id] = Counter(tokens)

    def get_documents(self, term: str) -> list[int]: #get the set of documents for a given token, and return them as a list, sorted in ascending order by document ID.
        token = term.lower()
        doc_ids = self.index.get(token, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) == 0:
            return 0 
        if len(tokenized_term) > 1: 
            raise ValueError("Term has multiple tokens")
        return self.term_frequncies[doc_id][tokenized_term[0]]
    
    
    def build(self) -> None: #It should iterate over all the movies and add them to both the index and the docmap.
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, doc_description)
            self.docmap[doc_id] = movie
        
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


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

            
