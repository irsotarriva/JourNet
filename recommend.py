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
from fastapi import APIRouter, HTTPException, Depends, status
import server
import papers
from loggin import get_current_user

# Load environment variables once
load_dotenv()

router = APIRouter()
@router.get("/", response_model=list[papers.PaperResponse])
def recommend_papers(current_user=Depends(get_current_user), top_k: int = 10) -> list[papers.PaperResponse]:
    #get the max id in supabase
    supabase: Client = get_supabase_client()
    response = supabase.table("Papers").select("id").order("id", desc=True).limit(1).execute()
    max_id = 0
    if response.data:
        max_id = response.data[0]["id"]
    user_id = current_user["id"]
    recommended_papers = get_recommendation(user_id, top_k=top_k, max_id=max_id)
    papers_l = []
    for paperid in recommended_papers:
        try:
            paper = papers.get_paper(int(paperid))
            if paper:
                papers_l.append(paper)
        except Exception as e:
            print("Warning: Failed fetching paper ID", paperid, ":", str(e))
    #pad with random papers if necessary
    while len(papers_l) < top_k:
        import random
        random_id = random.randint(1, max_id)
        try:
            paper = papers.get_paper(int(random_id))
            if paper and paper not in papers_l:
                papers_l.append(paper)
        except Exception as e:
            print("Warning: Failed fetching paper ID", random_id, ":", str(e))
    print("Recommended Papers IDs:", recommended_papers)
    return papers_l

@router.get("/search/", response_model=list[papers.PaperResponse])
def search_recommendations(query: str, current_user=Depends(get_current_user), top_k: int = 5) -> list[papers.PaperResponse]:
    print("Search Query:", query)
    #get the max id in supabase
    supabase: Client = get_supabase_client()
    response = supabase.table("Papers").select("id").order("id", desc=True).limit(1).execute()
    max_id = 0
    if response.data:
        max_id = response.data[0]["id"]
    user_id = current_user["id"]
    print("User ID:", user_id)
    recommended_papers = search_papers(user_id, query, top_k, max_id=max_id)
    print("Search Recommended Papers IDs:", recommended_papers)
    papers_l = []
    for paperid in recommended_papers:
        try:
            paper = papers.get_paper(int(paperid))
            print("Fetched Paper:", paper)
            if paper:
                papers_l.append(paper)
        except Exception as e:
            print("Warning: Failed fetching paper ID", paperid, ":", str(e))
    #pad with random papers if necessary
    while len(papers_l) < top_k:
        import random
        random_id = random.randint(1, max_id)
        try:
            paper = papers.get_paper(int(random_id))
            if paper and paper not in papers_l:
                papers_l.append(paper)
        except Exception as e:
            print("Warning: Failed fetching paper ID", random_id, ":", str(e))
    return papers_l

@router.get("/friends/", response_model=list[papers.PaperResponse])
def find_friends(paper_id: int, top_k: int = 5) -> list[papers.PaperResponse]:
    #get the max id in supabase
    supabase: Client = get_supabase_client()
    response = supabase.table("Papers").select("id").order("id", desc=True).limit(1).execute()
    max_id = 0
    if response.data:
        max_id = response.data[0]["id"]
    recommendation_engine = RecommendationEngine()
    paper_vector = recommendation_engine.get_vector_by_paper_id(paper_id)
    if paper_vector is [] or paper_vector is None:
        print("Paper Vector is empty or None. Generating random friends.")
        friends = []
        import random
        while len(friends) < top_k:
            random_id = random.randint(1, max_id)
            if random_id not in friends:
                try:
                    paper = papers.get_paper(int(random_id))
                    if paper:
                        friends.append(paper)
                except Exception as e:
                    print("Warning: Failed fetching paper ID", random_id, ":", str(e))
        #return list of papers
        return friends
    print("Paper Vector:", paper_vector)
    recommended_papers = recommendation_engine.get_recommendation(
        positive_embeddings=[paper_vector],
        negative_embeddings=[],
        top_k=top_k,
        filter=qdrant_models.Filter(
            must = [
                qdrant_models.FieldCondition(
                    key="supaIndex",
                    range =qdrant_models.Range(
                        gte=1,
                        lte=int(max_id)
                    )
                )
            ]
        )
    )
    print("Friendly Papers IDs:", recommended_papers)
    papers_l = []
    for paperid in recommended_papers:
        try:
            paper = papers.get_paper(int(paperid))
            print("My friend is:", paperid)
            print("Fetched Paper:", paper)
            if paper:
                papers_l.append(paper)
        except Exception as e:
            print("Warning: Failed fetching paper ID", paperid, ":", str(e))
    #pad with random papers if necessary
    while len(papers_l) < top_k:
        import random
        random_id = random.randint(1, max_id)
        try:
            paper = papers.get_paper(int(random_id))
            if paper and paper not in papers_l:
                papers_l.append(paper)
        except Exception as e:
            print("Warning: Failed fetching paper ID", random_id, ":", str(e))
    return papers_l

def _get_user_history_embeddings(user_id: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    @brief Fetch user history from Supabase and get positive and negative embeddings.
    @param user_id: ID of the user to fetch history for.
    @return: Tuple of lists containing positive and negative embeddings.
    @details The user history is fetched from the Supabase database, and the recommendation engine is used to suggest papers.
    Suggestions are based on a weighted combination of the embedding vectors of the papers the user has interacted with. The weights are as follows:
    - +1 per comment on a paper
    - +2 per appearance in user reading list
    - +3 for 5-star ratings
    - +2 for 4-star ratings
    - +1 for 3-star ratings
    - -1 for 1-star ratings
    """
    supabase: Client = get_supabase_client()
    # find all the user comments
    paper_ids_associated_with_comments = supabase.table("Discussion").select("articleId").eq("userId", user_id).execute()
    #find all the papers in the user reading list
    paper_ids_in_reading_list = supabase.table("ReadingList").select("paperid").eq("userid", user_id).execute()
    #find all the user ratings
    user_ratings = supabase.table("Ratings").select("paperid", "rating").eq("userid", user_id).execute()
    #make a dictionary to store the vectors and weights
    seen_paper_ids = set()
    paper_embebdings = {}
    paper_weights = {}
    recommendation_engine = RecommendationEngine()
    for paper in paper_ids_associated_with_comments.data:
        paper_id = paper["articleId"]
        if paper_id not in seen_paper_ids:
            vector = recommendation_engine.get_vector_by_paper_id(paper_id)
            if vector is not None:
                paper_embebdings[paper_id] = vector
                paper_weights[paper_id] = paper_weights.get(paper_id, 0) + 1
                seen_paper_ids.add(paper_id)
    for reading_list in paper_ids_in_reading_list.data:
        paper_ids = reading_list.get("paperid") or []
        for paper_id in paper_ids:
            if paper_id not in seen_paper_ids:
                vector = recommendation_engine.get_vector_by_paper_id(paper_id)
                if vector is not None:
                    paper_embebdings[paper_id] = vector
                    paper_weights[paper_id] = paper_weights.get(paper_id, 0) + 2
                    seen_paper_ids.add(paper_id)
    for rating in user_ratings.data:
        paper_id = rating["paperid"]
        rating_value = rating["rating"]
        if paper_id not in seen_paper_ids:
            vector = recommendation_engine.get_vector_by_paper_id(paper_id)
            if vector is not None:
                paper_embebdings[paper_id] = vector
                if rating_value == 5:
                    paper_weights[paper_id] = paper_weights.get(paper_id, 0) + 3
                elif rating_value == 4:
                    paper_weights[paper_id] = paper_weights.get(paper_id, 0) + 2
                elif rating_value == 3:
                    paper_weights[paper_id] = paper_weights.get(paper_id, 0) + 1
                elif rating_value == 1:
                    paper_weights[paper_id] = paper_weights.get(paper_id, 0) - 1
                seen_paper_ids.add(paper_id)
    # Prepare positive and negative embeddings
    positive_embeddings = []
    negative_embeddings = []
    for paper_id, weight in paper_weights.items():
        embedding = paper_embebdings[paper_id]
        if weight > 0:
            for _ in range(weight):
                positive_embeddings.append(embedding)
        elif weight < 0:
            for _ in range(-weight):
                negative_embeddings.append(embedding)
    return positive_embeddings, negative_embeddings

def get_recommendation(user_id: int, top_k: int = 10, max_id: int = 5000) -> list[int]:
    """
    @brief Get paper recommendations for a user based on their history. 
    @param user_id: ID of the user to get recommendations for.
    @return: List of recommended paper IDs.
    """
    recommendation_engine = RecommendationEngine()
    positive_embeddings, negative_embeddings = _get_user_history_embeddings(user_id)
    filter = qdrant_models.Filter(
        must = [
            qdrant_models.FieldCondition(
                key="supaIndex",
                range =qdrant_models.Range(
                    gte=1,
                    lte=int(max_id)
                ),
            )
        ]
    )
    # Get recommendations
    recommended_papers = recommendation_engine.get_recommendation(
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
        top_k=top_k,
        filter=filter
    )
    return recommended_papers
    return recommended_papers

def _get_exact_matches(query: str, limit: int = 10) -> list[int]:
    """
    @brief Get exact matches for a query string in the Papers table, sorted by comment count.
    @param query: Search query string.
    @param limit: Max number of matches to return.
    @return: List of paper IDs.
    """
    supabase: Client = get_supabase_client()
    # Simple sanitization/formatting for ilike
    # We want matches in title OR abstract
    filter_str = f"title.ilike.%{query}%,abstract.ilike.%{query}%"
    
    # Fetch papers (fetch more than limit to allow sorting)
    try:
        response = supabase.table("Papers").select("id").or_(filter_str).limit(50).execute()
        papers_data = response.data
    except Exception as e:
        print(f"Error fetching exact matches: {e}")
        return []

    if not papers_data:
        return []

    paper_ids = [p['id'] for p in papers_data]

    # Fetch discussion counts for these papers
    try:
        response_comments = supabase.table("Discussion").select("articleId").in_("articleId", paper_ids).execute()
        comment_counts = {}
        for c in response_comments.data:
            aid = c['articleId']
            comment_counts[aid] = comment_counts.get(aid, 0) + 1
        
        # Sort paper_ids by count desc
        paper_ids.sort(key=lambda pid: comment_counts.get(pid, 0), reverse=True)
    except Exception as e:
        print(f"Error fetching exact match comments: {e}")
        # Return unsorted or partially sorted if comments fail
    
    return paper_ids[:limit]

def search_papers(user_id: int, query: str, top_k: int = 5, max_id: int = 5000) -> list[int]:
    """
    @brief Search for papers using a hybrid approach: Exact matches first, then vector similarity.
    @param user_id: ID of the user.
    @param query: Search query string.
    @param top_k: Total number of results desired.
    @return: List of recommended paper IDs.
    """
    # 1. Get exact matches (Top 4 mostly related by comments)
    exact_matches = _get_exact_matches(query, limit=4)
    exact_ids_set = set(exact_matches)
    
    # 2. Fill the rest with recommendation engine
    recommendation_engine = RecommendationEngine()
    query_embedding = recommendation_engine.embed_text(query)
    
    positive_embeddings, negative_embeddings = _get_user_history_embeddings(user_id)
    positive_embeddings.append(query_embedding)
    
    remaining_slots = max(0, top_k - len(exact_matches))
    # If we need more, fetch a bit extra to handle duplicates
    vector_k = remaining_slots + len(exact_matches) + 2
    
    # Use vector search to find conceptually similar papers
    # We still use the title/abstract match as a SHOULD filter to boost relevance, 
    # but the primary goal is filling the slots with relevant content
    filter = qdrant_models.Filter(
        should = [
            qdrant_models.FieldCondition(
                key="title",
                match=qdrant_models.MatchPhrase(phrase=query)
            ),
            qdrant_models.FieldCondition(
                key="authors",
                match=qdrant_models.MatchPhrase(phrase=query)
            ),
            qdrant_models.FieldCondition(
                key="abstract",
                match=qdrant_models.MatchPhrase(phrase=query)
            )
        ],
        must = [
            qdrant_models.FieldCondition(
                key="supaIndex",
                range =qdrant_models.Range(
                    gte=1,
                    lte=int(max_id)
                )
            )
        ]
    )
    
    recommended_papers = recommendation_engine.get_recommendation(
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
        top_k=vector_k,
        filter=filter
    )
    
    # Combine results
    final_results = []
    # Add exact matches
    final_results.extend(exact_matches)
    
    # Add vector matches if NOT in exact matches
    for pid in recommended_papers:
        if pid not in exact_ids_set:
            final_results.append(pid)
            if len(final_results) >= top_k:
                break
                
    return final_results


class RecommendationEngine:
    qdrant_client = None
    collection_name = "arxiv_papers"
    embModel = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    def __init__(self):
        url: str = os.environ.get("QUADRANT_URL")
        key: str = os.environ.get("QUADRANT_KEY")
        if not self.qdrant_client:
            self.qdrant_client = QdrantClient(url=url, api_key=key)
    """
    def get_paper_by_id(self, paper_id: int) -> models.Paper:
        result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must = [
                    qdrant_models.FieldCondition(
                        key="supaIndex",
                        range =qdrant_models.Range(
                            gte=int(paper_id),
                            lt=int(paper_id + 1)
                        )
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
    """
    def get_vector_by_paper_id(self, paper_id: int) -> np.ndarray:
        results = self.qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=[id for id in range(paper_id-4, paper_id+6) if id > 0],  #look around +-5 IDs
        )
        for result in results:
            supa_index = result.payload.get("supaIndex", None)
            if supa_index == paper_id:
                vector = result.vector
                return np.array(vector)
        return None

    def get_recommendation(self, positive_embeddings: list[np.ndarray] = [], negative_embeddings: list[np.ndarray] = [], top_k: int = 5, filter: qdrant_models.Filter = None) -> list[int]:
        if filter is None:
            filter = qdrant_models.Filter(
                must_not = [
                    qdrant_models.IsNullCondition(
                        is_null=qdrant_models.PayloadField(key = "supaIndex")
                    )
                ]
            )
        if not positive_embeddings:
            positive_embeddings = [self.embed_text("Science")]
        if not negative_embeddings:
            negative_embeddings = [self.embed_text("Pseudoscience")]
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
        paper_ids = []
        for result in results:
            paper_ids.append(result.payload.get("supaIndex", None))
        return paper_ids

    """
    def add_supabase_index_to_payload(self, batch_size: int = 256):
        #Iterate over all points in the Qdrant collection, compute the
        #corresponding Supabase index, and add it to the payload as `supaIndex`.

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
    """
    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.embModel.encode(text)
        return embedding

if __name__ == "__main__":
    engine = RecommendationEngine()
    #index the supaIndex
    engine.qdrant_client.create_payload_index(
        collection_name=engine.collection_name,
        field_name="supaIndex",
        field_schema="integer",
    )

    """
    engine = RecommendationEngine()
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
    """
