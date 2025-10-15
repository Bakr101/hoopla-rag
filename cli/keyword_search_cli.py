#!/usr/bin/env python3


import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.keyword_search import (search_command, tf_command,build_command)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()