import argparse
from dotenv import load_dotenv
from lib.hybrid_search import rrf_search
from lib.search_utils import generate_content
def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a document"
    )
    summarize_parser.add_argument("query", type=str, help="Query to search and summarize")
    summarize_parser.add_argument("limit", type=int, help="Limit the number of results", default=5, nargs="?")
    citation_parser = subparsers.add_parser(
        "citations", help="LLM answer after performing search with citation from the documents provided to it"
    )
    citation_parser.add_argument("query", type=str, help="Query to search and provide a citation")
    citation_parser.add_argument("limit", type=int, help="Limit the number of results", default=5, nargs="?")

    question_parser = subparsers.add_parser(
        "question", help="LLM answer after performing search with answer to the user's question from the documents provided to it"
    )
    question_parser.add_argument("query", type=str, help="Query to search and provide an answer to the user's question")
    question_parser.add_argument("limit", type=int, help="Limit the number of results", default=5, nargs="?")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rrf_search_result = rrf_search(query, limit=5, evaluate=False)
            results = rrf_search_result["results"]
            titles_documents = []
            titles_found = ""
            for result in results:
                titles_documents.append(f"{result['title']}\n{result['document']}")
                titles_found += f"- {result["title"]}\n"
            titles_documents_text = "\n".join(titles_documents)
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{titles_documents_text}

Provide a comprehensive answer that addresses the query:"""
            load_dotenv()
            answer = generate_content(prompt)
            print("Search Results:")
            print(titles_found)
            print("RAG Response:")
            print(answer)

        case "summarize":
            query = args.query
            limit = args.limit
            rrf_search_result = rrf_search(query, limit=limit, evaluate=False)
            results = rrf_search_result["results"]
            titles_documents = []
            titles_found = ""
            for result in results:
                titles_documents.append(f"{result['title']}\n{result['document']}")
                titles_found += f"- {result["title"]}\n"
            titles_documents_text = "\n".join(titles_documents)
            prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
            load_dotenv()
            answer = generate_content(prompt)
            print("Search Results:")
            print(titles_found)
            print("Summarized Response:")
            print(answer)
        case "citations":
            query = args.query
            limit = args.limit
            rrf_search_result = rrf_search(query, limit=limit, evaluate=False)
            results = rrf_search_result["results"]
            titles_documents = []
            titles_found = ""
            for result in results:
                titles_documents.append(f"{result['title']}\n{result['document']}")
                titles_found += f"- {result["title"]}\n"
            titles_documents_text = "\n".join(titles_documents)
            prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{titles_documents_text}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
            load_dotenv()
            answer = generate_content(prompt)
            print("Search Results:")
            print(titles_found)
            print("LLM Answer:")
            print(answer)
        case "question":
            query = args.query
            limit = args.limit
            rrf_search_result = rrf_search(query, limit=limit, evaluate=False)
            results = rrf_search_result["results"]
            titles_documents = []
            titles_found = ""
            for result in results:
                titles_documents.append(f"{result['title']}\n{result['document']}")
                titles_found += f"- {result["title"]}\n"
            titles_documents_text = "\n".join(titles_documents)
            
            load_dotenv()
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{titles_documents_text}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
            answer = generate_content(prompt)
            print("Search Results:")
            print(titles_found)
            print("LLM Answer:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()