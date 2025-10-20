#!/usr/bin/env python3


import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.keyword_search import (search_command, tf_command,build_command, idf_command, bm25_idf_command, tfidf_command, bm25_tf_command, bm25search_command)
from lib.search_utils import (BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score")
    bm25_idf_parser.add_argument("term", type=str, help="Term")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term")
    bm25_tf_parser.add_argument("k1", type=float, help="Tunable BM25 K1 parameter", default=BM25_K1, nargs="?")
    bm25_tf_parser.add_argument("b", type=float, help="Tunable BM25 B parameter", default=BM25_B, nargs="?")

    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using BM25")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("limit", type=int, help="Limit the number of results", default=DEFAULT_SEARCH_LIMIT, nargs="?")
    args = parser.parse_args()
    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for index, result in enumerate(results, 1):
                print(f"{index}. {result['title']} {result['id']}")
        case "build":
            print("Building index...")
            build_command()
        case "tf":
            result = tf_command(args.doc_id, args.term)
            print(f"Term frequency: {result}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            bm25 = bm25search_command(args.query, args.limit)
            for dic in bm25:
                print(f"({dic['id']}) {dic['title']} {dic['score']:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()