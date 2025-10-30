import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (load_movies, DEFAULT_ALPHA, format_search_result, load_llm_client, GEMINI_FLASH_MODEL)
from .query_enhancment import (enhance_query, llm_rerank)


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


    def rrf_search(self, query, k=60, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        doc_id_to_scores = {}
        for idx, result in enumerate(bm25_results):
            doc_id = result["doc_id"] - 1
            doc = self.documents[doc_id]
            doc_id_to_scores[doc_id] = {
                "bm25_rank": idx + 1,
                "semantic_rank": None,
                "doc": doc,
            }
        for idx, result in enumerate(semantic_results):
            doc_id = result["doc_id"] - 1
            doc = self.documents[doc_id]
            if doc_id not in doc_id_to_scores:
                doc_id_to_scores[doc_id] = {
                    "semantic_rank": idx + 1,
                    "bm25_rank": None,
                    "doc": doc,
                }
            else:
                doc_id_to_scores[doc_id]["semantic_rank"] = idx + 1
        for doc_id, data in doc_id_to_scores.items():
            if data["bm25_rank"] is not None and data["semantic_rank"] is not None:
                data["rrf_score"] = rrf_score(data["bm25_rank"], k) + rrf_score(data["semantic_rank"], k)
            elif data["bm25_rank"] is not None:
                data["rrf_score"] = rrf_score(data["bm25_rank"], k)
            elif data["semantic_rank"] is not None:
                data["rrf_score"] = rrf_score(data["semantic_rank"], k)

        sorted_results = sorted(doc_id_to_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        return sorted_results[:limit]


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

def combine_search_results(bm25_results: list[dict], semantic_results: list[dict], alpha: float= DEFAULT_ALPHA):
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

def rrf_score(rank: int, k: int=60) -> float:
    return 1 / (k + rank)


    

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

def rrf_search(query, k=60, limit=5, method=None, rerank_method=None):
    movies = load_movies()
    original_query = query
    enhanced_query = None
    if rerank_method:
        new_limit = limit * 5
    if method:
        enhanced_query = enhance_query(query, method)
        query = enhanced_query
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, k, new_limit)[:limit]
    if rerank_method:
        results = llm_rerank(query, results, rerank_method)
    results = results[:limit]
    return {
        "query": query,
        "k": k,
        "limit": limit,
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": method,
        "results": results,
    }

