from typing import Optional
from .search_utils import (load_llm_client, GEMINI_FLASH_MODEL)
from dotenv import load_dotenv
import time
import json
import re
def spell_correct(query):
    load_dotenv()
    client = load_llm_client()
    
    content = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    generated_content = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=content
        )
        
    corrected_query = (generated_content.text or "").strip().strip('"').replace('Corrected: ', '').replace('"', '')
    return corrected_query if corrected_query else query

def rewrite_query(query):
    load_dotenv()
    client = load_llm_client()
    
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    generated_content = client.models.generate_content(
        model=GEMINI_FLASH_MODEL,
        contents=prompt
    )
    rewritten_query = (generated_content.text or "").strip().strip('"').replace('Rewritten query: ', '').replace('"', '')
    return rewritten_query if rewritten_query else query


def expand_query(query):
    load_dotenv()
    client = load_llm_client()
    
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    generated_content = client.models.generate_content(
        model=GEMINI_FLASH_MODEL,
        contents=prompt
    )
    expanded_query = (generated_content.text or "").strip().strip('"')
    return expanded_query if expanded_query else query

def enhance_query(query, method: Optional[str]=None):
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query


def individual_rerank(query, results):
    load_dotenv()
    client = load_llm_client()

    for result in results:
        doc = result[1]["doc"]
        title = doc.get("title", "")
        bm25_rank = result[1]["bm25_rank"]
        semantic_rank = result[1]["semantic_rank"]
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        generated_content = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt
        )
        score = (generated_content.text or "").strip().strip('"').replace('Score: ', '').replace('"', '')
        result[1]["rerank_score"] = float(score)
        time.sleep(2)
    sorted_results = sorted(results, key=lambda x: x[1]["rerank_score"], reverse=True)
    return sorted_results


def batch_rerank(query, results):
    load_dotenv()
    client = load_llm_client()
    doc_list_str = "\n".join([f"id: {result[0]}, title: {result[1]['doc']['title']}, description: {result[1]['doc']['description']}" for i, result in enumerate(results)])
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    generated_content = client.models.generate_content(
        model=GEMINI_FLASH_MODEL,
        contents=prompt
    )
    raw = (generated_content.text or "").strip()

    if not raw:
        raise ValueError("LLM returned empty response for reranking")

    # Extract the JSON array if the model added extra text
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        raw = m.group(0)

    try:
        reranked_ids = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Expected JSON list of IDs from LLM, got: {raw[:200]}") from e

    reranked_results = {}
     # Map doc_id -> (doc_id, data)
    id_to_result = {doc_id: (doc_id, data) for doc_id, data in results}

    # Build ordered list, keeping only those present
    ordered_results = [id_to_result[doc_id] for doc_id in reranked_ids if doc_id in id_to_result]

    return ordered_results
    
def llm_rerank(query, results, rerank_method):
    match rerank_method:
        case "individual":
            return individual_rerank(query, results)
        case "batch":
            return batch_rerank(query, results)
        case _:
            return results

