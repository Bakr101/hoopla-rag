import argparse
import mimetypes
import os
from dotenv import load_dotenv
from lib.search_utils import load_llm_client, GEMINI_FLASH_MODEL
from google import genai


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, help="Path to the image to describe")
    parser.add_argument(
        "--query", type=str, help="a text query to rewrite based on the image"
    )
    args = parser.parse_args()
    mime_type, _ = mimetypes.guess_type(args.image)
    mime_type = mime_type or "image/jpeg"

    image_file = open(args.image, "rb")
    image_data = image_file.read()
    image_file.close()

    load_dotenv()
    client = load_llm_client()
    prompt = f"""
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
    """
    parts = [
        args.query.strip(),
        genai.types.Part.from_bytes(mime_type=mime_type, data=image_data),
        prompt,
    ]
    response = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
