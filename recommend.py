#import server
import models

from pydantic import BaseModel
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

#server_instance = server.Server()
#app = server_instance.get_app()

class RecommendationEngine:
    qdrant_client = QdrantClient(
        url=os.getenv("QUADRANT_URL"),
        api_key=os.getenv("QUADRANT_KEY")
    )
    collection_name = "arxiv_papers"
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    def __init__(self):
        print(self.qdrant_client.get_collections())
    def get_paper_by_id(self, paper_id: str) -> models.Paper:
        result = self.qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=[paper_id],
            with_payload=True,
            with_vectors=True,
        )
        print("Scroll result:", result)
        """
        if not result:
            return None
        str_authors = result.payload.get("authors", "")
        author_list = []
        if str_authors:
            authors_split = str_authors.split(", ")
            for author_name in authors_split:
                author_list.append(models.Author(name=author_name))
        return models.Paper(
            uuid=result.id,
            title=result.payload.get("title", ""),
            abstract=result.payload.get("abstract", ""),
            authors=author_list,
            journal_ref=result.payload.get("journal_ref", ""),
            doi=result.payload.get("doi", ""),
            report_number=result.payload.get("report_number", ""),
            categories=result.payload.get("categories", "").split(", "),
            paper_license=result.payload.get("paper_license", ""),
            comments=result.payload.get("comments", "")
        )
        """
        return None
    def get_recommendation_for_paper(self, embedding: np.ndarray, top_k: int = 5) -> list[models.Paper]:
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=top_k
        )
        recommendations = []
        for result in search_results:
            str_authors = result.payload.get("authors", "")
            author_list = []
            if str_authors:
                authors_split = str_authors.split(", ")
                for author_name in authors_split:
                    author_list.append(models.Author(name=author_name))
            recommendations.append(models.Paper(
                uuid=result.id,
                title=result.payload.get("title", ""),
                abstract=result.payload.get("abstract", ""),
                authors=author_list,
                journal_ref=result.payload.get("journal_ref", ""),
                doi=result.payload.get("doi", ""),
                report_number=result.payload.get("report_number", ""),
                categories=result.payload.get("categories", "").split(", "),
                paper_license=result.payload.get("paper_license", ""),
                comments=result.payload.get("comments", "")
            ))
        return recommendations
    def get_recommendation(self, positive_embeddings: list[np.ndarray] = [], negative_embeddings: list[np.ndarray] = [], top_k: int = 5, filter: dict = None) -> list[models.Paper]:
        filter = qdrant_models.Filter(
            must = [
                qdrant_models.FieldCondition(
                    key=filed_name,
                    match=qdrant_models.MatchValue(value=field_value)
                ) for filed_name, field_value in filter.items()
            ]
        ) if filter else None
        recomment_query = qdrant_models.QueryRequest(
            query = qdrant_models.RecommendQuery(
                recommend = qdrant_models.RecommendInput(
                    positive = [emb.tolist() for emb in positive_embeddings],
                    negative = [emb.tolist() for emb in negative_embeddings],
                    strategy = "best_score"
                )
            ),
            limit = top_k,
            filter = filter
        )
        search_results = self.qdrant_client.recommend(
            collection_name=self.collection_name,
            query=recomment_query
        )
        recommendations = []
        for result in search_results:
            str_authors = result.payload.get("authors", "")
            author_list = []
            if str_authors:
                authors_split = str_authors.split(", ")
                for author_name in authors_split:
                    author_list.append(models.Author(name=author_name))
            recommendations.append(models.Paper(
                uuid=result.id,
                title=result.payload.get("title", ""),
                abstract=result.payload.get("abstract", ""),
                authors=author_list,
                journal_ref=result.payload.get("journal_ref", ""),
                doi=result.payload.get("doi", ""),
                report_number=result.payload.get("report_number", ""),
                categories=result.payload.get("categories", "").split(", "),
                paper_license=result.payload.get("paper_license", ""),
                comments=result.payload.get("comments", "")
            ))
        return recommendations
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

if __name__ == "__main__":
    engine = RecommendationEngine()
    """
    # Example usage
    sample_positive_text1 = "Deep learning techniques for natural language processing"
    sample_positive_text2 = "Advancements in computer vision using convolutional neural networks"
    sample_negative_text = "Statistical methods in classical physics"
    print("Generating embeddings")
    pos_emb1 = engine.embed_text(sample_positive_text1)
    pos_emb2 = engine.embed_text(sample_positive_text2)
    neg_emb = engine.embed_text(sample_negative_text)
    print("Fetching recommendations")

    recommendations = engine.get_recommendation(
        positive_embeddings=[pos_emb1, pos_emb2],
        negative_embeddings=[neg_emb],
        top_k=5,
        filter={"categories": "cs.CL"}
    )
    for paper_rec in recommendations:
        print(f"Paper ID: {paper_rec.uuid}, Title: {paper_rec.title}")
    """
    paper_id = 14
    paper_rec = engine.get_paper_by_id(paper_id)
    if paper_rec:
        print(f"Paper ID: {paper_rec.uuid}, Title: {paper_rec.title}", f"Authors: {[author.name for author in paper_rec.authors]}")
