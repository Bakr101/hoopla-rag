#!/usr/bin/env python3

import argparse
from lib.semantic_search import (embed_query_text, verify_model, embed_text, verify_embeddings, search, chunk, semantic_chunk, embed_chunks, semantic_chunk_text, search_chunks)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed a text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")
    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, help="Limit the number of results", default=5, nargs="?")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, help="Chunk size", default=200, nargs="?")
    chunk_parser.add_argument("--overlap", type=int, help="Overlap between chunks in words", default=0, nargs="?")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk a text")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to semantic chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, help="Chunk size in sentences", default=4, nargs="?")
    semantic_chunk_parser.add_argument("--overlap", type=int, help="Overlap between chunks in sentences", default=0, nargs="?")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate chunk embeddings from documents")
    
    search_chunks_parser = subparsers.add_parser("search_chunked", help="Search chunks")
    search_chunks_parser.add_argument("query", type=str, help="Search query")
    search_chunks_parser.add_argument("--limit", type=int, help="Limit the number of results", default=5, nargs="?")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunks(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()