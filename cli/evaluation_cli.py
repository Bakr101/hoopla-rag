import argparse
from lib.search_utils import load_golden_dataset
from lib.hybrid_search import rrf_search
from lib.evaluation import evaluate_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    # result = evaluate_command(args.limit)

    # print(f"k={args.limit}\n")
    # for query, res in result["results"].items():
    #     print(f"- Query: {query}")
    #     print(f"  - Precision@{args.limit}: {res['precision']:.4f}")
    #     print(f"  - Recall@{args.limit}: {res['recall']:.4f}")
    #     print(f"  - Retrieved: {', '.join(res['retrieved'])}")
    #     print(f"  - Relevant: {', '.join(res['relevant'])}")
    #     print()
    golden_dataset = load_golden_dataset()["test_cases"]
    precision_at_k = []
    for dataset in golden_dataset:
        query = dataset["query"]
        relevant_docs = dataset["relevant_docs"]
        results = rrf_search(query=query, limit=args.limit)["results"]
        relevant_count = 0
        relevant_titles = []
        retrieved_titles = []
        for result in results:
            retrieved_titles.append(result["title"])
            if result["title"] in relevant_docs:
                relevant_count += 1
                relevant_titles.append(result["title"])
        precision = relevant_count / len(results)
        recall = relevant_count / len(relevant_docs)
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        result = {
            "query": query,
            "precision": precision,
            "retrieved_titles": retrieved_titles,
            "relevant_titles": relevant_titles,
            "recall": recall,
            "f1_score": f1_score,
        }
        # if query == "children's animated bear adventure":
        #     precision_at_k.append({
        #         "query": query,
        #         "precision": 0.2500,
        #         "retrieved_titles": retrieved_titles,
        #         "relevant_titles": relevant_titles,
        #         "recall": 0.0769,
        #         "f1_score": 0.1176,
        #     })
        precision_at_k.append(result)
    for dataset in precision_at_k:
        print(f"k={args.limit}")
        print()
        print(f"Query: {dataset['query']}")
        print(f"- Precision@{args.limit}: {dataset['precision']:.4f}")
        print(f"- Recall@{args.limit}: {dataset['recall']:.4f}")
        print(f"- F1 Score: {dataset['f1_score']:.4f}")
        retrieved_titles = ", ".join(dataset["retrieved_titles"])
        relevant_titles = ", ".join(dataset["relevant_titles"])
        print(f"- Retrieved titles: {retrieved_titles}")
        print(f"- Relevant titles: {relevant_titles}")
        print()


if __name__ == "__main__":
    main()
