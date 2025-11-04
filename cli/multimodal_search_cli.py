import argparse
from lib.multimodal_search import verify_image_embedding, search_with_image

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify the image embedding")
    verify_image_embedding_parser.add_argument("image", type=str, help="Path to the image to embed")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search with image")
    image_search_parser.add_argument("image", type=str, help="Path to the image to search with")

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            search_with_image(args.image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()