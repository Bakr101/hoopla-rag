import json
from sentence_transformers import SentenceTransformer
from lib.search_utils import (format_search_result, load_movies, CACHE_PATH)
import numpy as np
import os
import re

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.documents_map = {}
        self.embeddings_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Text cannot be None or empty")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        m_descriptions = []
        for doc in documents:
            self.documents_map[doc["id"]] = doc
            m_descriptions.append(f"{doc['title']} {doc['description']}")
        self.embeddings = self.model.encode(m_descriptions, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        print(f"Embeddings saved to {self.embeddings_path}")
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings 
        else:
            return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            doc_id = list(self.documents_map.keys())[i]
            similarities.append([similarity, self.documents_map[doc_id]])
        similarities.sort(key=lambda x: x[0], reverse=True)
        similarities = similarities[:limit]
        results = []
        for similarity, doc in similarities:
            m = {
                "score": similarity,
                "title": doc["title"],
                "description": doc["description"]
            }
            results.append(m)
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_PATH, "chunk_metadata.json")
    
    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        chunks: list[str] = []
        chunk_metadata: list[dict] = []
        for doc in documents:
            self.documents_map[doc["id"]] = doc
            description = doc['description']
            if description is None:
                continue
            semantic_chunks = semantic_chunk(description, chunk_size=4, overlap=1)
            chunks.extend(semantic_chunks)
            for idx in range(len(semantic_chunks)):
                chunk_metadata.append({
                    "movie_idx": doc["id"],
                    "chunk_idx": idx,
                    "total_chunks": len(semantic_chunks)
                })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        print(f"Chunk embeddings saved to {self.chunk_embeddings_path}")
        print(f"Chunk metadata saved to {self.chunk_metadata_path}")
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int=10):
        query_embedding = self.model.encode([query])
        chunk_scores: list[dict] = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append({
                "score": score,
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
            })
        m_scores = {}
        for score in chunk_scores:
            if score["movie_idx"] not in m_scores or score["score"] > m_scores[score["movie_idx"]]:
                m_scores[score["movie_idx"]] = score["score"]
        m_scores = sorted(m_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results = []
        for m in m_scores:
            m_idx = m[0]
            m_score = m[1][0]
            print(f"Movie ID: {m_idx}, Score: {m_score}")
            results.append(format_search_result(doc_id=m_idx, title=self.documents_map[m_idx]["title"], document=self.documents_map[m_idx]["description"], score=m_score))
        return results



def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

    
def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding

def search(query: str, limit: int = 5):
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)
    for index, result in enumerate(results, 0):
        print(f"{index + 1}. {result['title']} ({result['score']})")
        print(f"    {result['description']}")


def chunk(text: str, chunk_size: int = 200, overlap: int = 0):
    words = text.split(" ")
    chunks = []
    print(f"Chunking {len(text)} characters")
    for i in range(0, len(words), chunk_size):
        if overlap > 0 and len(chunks) > 0:
            prev_chunk = chunks[-1]
            overlap_words = prev_chunk.split(" ")[-overlap:]
            current_chunk = " ".join(overlap_words + words[i:i+chunk_size])
            chunks.append(current_chunk)
        else:    
            current_chunk = " ".join(words[i:i+chunk_size])
            chunks.append(current_chunk)
    for i in range(0, len(chunks)):
        print(f"{i + 1}. {chunks[i]}")
    return chunks

def semantic_chunk(text: str, chunk_size: int = 4, overlap: int = 0):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and (sentences[0].endswith(".") or sentences[0].endswith("!") or sentences[0].endswith("?")):
        sentences = [text]
    else:
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence == "":
                sentences.remove(sentence)
    chunks = []
    for i in range(0, len(sentences) - overlap, chunk_size - overlap):
        if overlap > 0 and len(chunks) > 0:
            prev_chunk = chunks[-1]
            overlap_sentences = re.split(r"(?<=[.!?])\s+", prev_chunk)[-overlap:]
            current_chunk = " ".join(overlap_sentences + sentences[i:i+chunk_size])
            chunks.append(current_chunk)
        else:
            current_chunk = " ".join(sentences[i:i+chunk_size])
            chunks.append(current_chunk)
    return chunks

def semantic_chunk_text(text: str, chunk_size: int = 4, overlap: int = 0):
    print(f"Semantically chunking {len(text)} characters")
    chunks = semantic_chunk(text, chunk_size, overlap)
    for i in range(0, len(chunks)):
        print(f"{i + 1}. {chunks[i]}")
    return chunks

def search_chunks(query: str, limit: int = 5):
    chunked_semantic_search = ChunkedSemanticSearch()
    movies = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    results = chunked_semantic_search.search_chunks(query, limit)
    for index, result in enumerate(results, 0):
        print(f"{index + 1}. {result['title']} ({result['score']:0.4f})")
        description = result["document"][:100]
        print(f"    {description}...")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_chunks():
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(chunked_semantic_search.chunk_embeddings)} chunked embeddings")