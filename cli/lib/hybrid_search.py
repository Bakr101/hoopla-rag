import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (load_movies, DEFAULT_ALPHA, format_search_result)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5) -> list[dict]:
        
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = combine_search_results(bm25_results, semantic_results, alpha)
        return combined_results[:limit]


    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return scores
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        scores = [1.0] * len(scores)
        # for score in scores:
        #     print(f"* {score:0.4f}")
        return scores
    scores = [(score - min_score) / (max_score - min_score) for score in scores]
    # for score in scores:
    #     print(f"* {score:0.4f}")
    return scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])
    normalized_scores: list[float] = normalize(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized_scores[i]
    return results


def hybrid_score(bm25_score, semantic_score, alpha=DEFAULT_ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def combine_search_results(bm25_results: list[dict], semantic_results: list[dict], alpha: float= DEFAULT_ALPHA) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["doc_id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "semantic_score": 0.0,
                "bm25_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]
    for result in semantic_normalized:
        doc_id = result["doc_id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]
    
    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)
    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

def weighted_search(query, alpha, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.weighted_search(query, alpha, limit)
    
    return {
        "query": query,
        "alpha": alpha,
        "limit": limit,
        "results": results,
    }
