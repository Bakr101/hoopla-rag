from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents: list[dict] = None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]
    
    def search_with_image(self, image_path: str):
        image = Image.open(image_path)
        image_embedding = self.embed_image(image_path)
        results = []
        for i, embedding in enumerate(self.embeddings,):
            similarity = cosine_similarity(image_embedding, embedding)
            results.append({
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "similarity": similarity,
                "doc_id": self.documents[i]["id"],
            })
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        results = results[:5]
        return results
        
        

def verify_image_embedding(image_path: str):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(image_path)
    print(f"Image: {image_path}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
    
def search_with_image(image_path: str):
    movies = load_movies()
    multimodal_search = MultimodalSearch(documents=movies)
    results = multimodal_search.search_with_image(image_path)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
        print(f"   {result['description'][:100]}...")
    