#import server
import models
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from connect_sql import get_supabase_client
from supabase import Client

import server

# Load environment variables once
load_dotenv()

server_instance = server.Server()
app = server_instance.get_app()



class RecommendationEngine:
    qdrant_client = None
    collection_name = "arxiv_papers"
    embModel = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    def __init__(self):
        url: str = os.environ.get("QUADRANT_URL")
        key: str = os.environ.get("QUADRANT_KEY")
        if not self.qdrant_client:
            self.qdrant_client = QdrantClient(url=url, api_key=key)
    def get_paper_by_id(self, paper_id: int) -> models.Paper:
        result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="supaIndex",
                        match=MatchValue(value=str(paper_id))
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False
        )
        if not result:
            return None
        result = result[0][0]
        str_authors = result.payload.get("authors", "")
        author_list = []
        if str_authors:
            authors_split = str_authors.split(", ")
            for author_name in authors_split:
                author_list.append(models.Author(name=author_name))
        return models.Paper(
            id=result.payload.get("supaIndex", ""),
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
    def get_vector_by_paper_id(self, paper_id: int) -> np.ndarray:
        result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="supaIndex",
                        match=MatchValue(value=str(paper_id))
                    )
                ]
            ),
            with_payload=False,
            with_vectors=True
        )
        if not result:
            return None
        result = result[0][0]
        return np.array(result.vector)
    def get_recommendation(self, positive_embeddings: list[np.ndarray] = [], negative_embeddings: list[np.ndarray] = [], top_k: int = 5, filter: qdrant_models.Filter = None) -> list[models.Paper]:
        if filter is None:
            filter = qdrant_models.Filter(
                must_not = [
                    qdrant_models.IsNullCondition(
                        is_null=qdrant_models.PayloadField(key = "supaIndex")
                    )
                ]
            )
        recommendation = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query = qdrant_models.RecommendQuery(
                recommend = qdrant_models.RecommendInput(
                    positive = [emb.tolist() for emb in positive_embeddings],
                    negative = [emb.tolist() for emb in negative_embeddings],
                    strategy = "best_score"
                )
            ),
            limit = top_k,
            query_filter = filter
        )
        results = recommendation.points
        papers = []
        for result in results:
            str_authors = result.payload.get("authors", "")
            author_list = []
            if str_authors:
                authors_split = str_authors.split(", ")
                for author_name in authors_split:
                    author_list.append(models.Author(name=author_name))
            papers.append(models.Paper(
                id=result.payload.get("supaIndex", ""),
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
        return papers
    def add_supabase_index_to_payload(self, batch_size: int = 256):
        """
        Iterate over all points in the Qdrant collection, compute the
        corresponding Supabase index, and add it to the payload as `supaIndex`.
        """

        #next_offset = None
        #total_updated = 0
        #start from 20K to skipp already updated
        next_offset = 20000
        total_updated = 20000

        while True:
            points, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                break

            for point in points:
                qdrant_id = point.id
                # Map Qdrant index -> Supabase index
                slip = 2  # Supabase index offset
                supa_index = qdrant_id + slip
                old_payload = point.payload or {}
                # Check with the supabase client to check if the article title at index supa_index matches. If it doesn't match try moving the slip by +1 or -1 and test again. If still not match trying +2 or -2 and so on up to +-n.
                n=5
                supabase: Client = get_supabase_client()
                response = supabase.table("Papers").select("title").eq("id", supa_index).execute()
                if response.data:
                    supabase_title = response.data[0]["title"]
                    qdrant_title = old_payload.get("title", "")
                    if supabase_title != qdrant_title:
                        found_match = False
                        for i in range(1, n+1):
                            # Check +i
                            response_plus = supabase.table("Papers").select("title").eq("id", supa_index + i).execute()
                            if response_plus.data and response_plus.data[0]["title"] == qdrant_title:
                                supa_index += i
                                found_match = True
                                break
                            # Check -i
                            response_minus = supabase.table("Papers").select("title").eq("id", supa_index - i).execute()
                            if response_minus.data and response_minus.data[0]["title"] == qdrant_title:
                                supa_index -= i
                                found_match = True
                                break
                        if not found_match:
                            print(f"⚠️ Could not find matching Supabase index for Qdrant ID {qdrant_id} with title '{qdrant_title}'")
                            #end the program with error
                            continue
                        #update slip
                        slip = supa_index - qdrant_id
                        print(f"🔄 Adjusted slip to {slip} starting from Qdrant ID {qdrant_id}")
                else:
                    print(f"⚠️ No Supabase entry found for index {supa_index}")
                    continue
                newPayload = old_payload.copy()
                newPayload["supaIndex"] = supa_index
                # Update payload only (no vectors touched)
                self.qdrant_client.set_payload(
                    collection_name=self.collection_name,
                    payload=newPayload,
                    points=[qdrant_id],
                )

                total_updated += 1

            if next_offset is None:
                break

        print(f"✅ Updated {total_updated} points with supaIndex")

        # Create payload index for fast filtering
        self.qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="supaIndex",
            field_schema="keyword",
        )

        print("⚡ Payload index created for `supaIndex`")
    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.embModel.encode(text)
        return embedding

if __name__ == "__main__":
    engine = RecommendationEngine()
    #engine.add_supabase_index_to_payload()

    # Example usage
    #sample_positive_text1 = "Deep learning techniques for natural language processing"
    sample_positive_text2 = "Advancements in computer vision using convolutional neural networks"
    sample_negative_text = "Statistical methods in classical physics"
    print("Generating embeddings")
    #pos_emb1 = engine.embed_text(sample_positive_text1)
    pos_emb2 = engine.embed_text(sample_positive_text2)
    neg_emb = engine.embed_text(sample_negative_text)
    print("Fetching recommendations")
    recommendations = engine.get_recommendation(
        positive_embeddings=[pos_emb2],
        negative_embeddings=[neg_emb],
        top_k=5
    )
    print("Recommendations:")
    for paper_rec in recommendations:
        print(f"Paper ID: {paper_rec.id}, Title: {paper_rec.title}")
    # Example with filter
    filter = qdrant_models.Filter(
        must = [
            qdrant_models.FieldCondition(
                key="title",
                match=qdrant_models.MatchPhrase(phrase="polymers")
            )
        ]
    )
    recommendations = engine.get_recommendation(
        positive_embeddings=[pos_emb2],
        negative_embeddings=[neg_emb],
        top_k=5,
        filter=filter
    )
    print("Filtered Recommendations (title contains 'polymers'):")
    for paper_rec in recommendations:
        print(f"Paper ID: {paper_rec.id}, Title: {paper_rec.title}")

    # Fetch paper by ID
    print("Fetching paper by ID 14")
    paper_id = 14
    paper_rec = engine.get_paper_by_id(paper_id)
    if paper_rec:
        print(f"Paper ID: {paper_rec.id}, Title: {paper_rec.title}", f"Authors: {[author.name for author in paper_rec.authors]}")
    print("Fetching vector by paper ID 14")
    vector = engine.get_vector_by_paper_id(paper_id)
    if vector is not None:
        print(f"Vector for Paper ID {paper_id}: {vector}")
