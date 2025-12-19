import server
from pydantic import BaseModel
import paper
import os
from qdrant_client import QdrantClient

server_instance = server.Server()
app = server_instance.get_app()

class RecommendationEngine:
    qdrant_client = QdrantClient(
        url="https://ab493be3-1f1f-4d1b-9f6d-150d5daca89d.sa-east-1-0.aws.cloud.qdrant.io:6333", 
        api_key=os.getenv("QDRANT_API_KEY")
    )

    def __init__(self):
        pass
    def get_recommendation_for_vector(embedding: list[float], top_k: int = 5):
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        collection_name = "arxiv_papers"

        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k
        )
        recommendations = []
        for result in search_results:
            str_authors = result.payload.get("authors", "")
            author_list = []
            if str_authors:
                authors_split = str_authors.split(", ")
                for author_name in authors_split:
                    author_list.append(paper.Author(name=author_name))
            recommendations.append(paper.Paper(
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