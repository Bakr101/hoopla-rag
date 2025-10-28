import argparse
from lib.hybrid_search import (
    normalize,
    weighted_search
)
from lib.search_utils import (DEFAULT_ALPHA)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Accepts a list of scores and prints the normalized scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, help="Dynamically controls the weighting between Semantic Search and Keyword search.", default=DEFAULT_ALPHA, nargs="?")
    weighted_search_parser.add_argument("--limit", type=int, help="Limit the number of results", default=5, nargs="?")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, help="Constant that controls how much more weight we give to higher-ranked results vs. lower-ranked ones Default is 60", default=60, nargs="?")
    rrf_search_parser.add_argument("--limit", type=int, help="Limit the number of results", default=5, nargs="?")

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            result = weighted_search(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "normalize":
            normalize(args.scores)
        case "rrf-search":
            result = rrf_search(args.query, args.k, args.limit)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()