import kagglehub
import numpy as np
import json
import time  # Added for retry mechanism
from tqdm.auto import tqdm
from connect_sql import get_supabase_client
from supabase import Client
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
import os
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance, PointStruct , PointVectors
# Qdrant configuration
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_KEY")
if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL and QDRANT_KEY must be set in environment variables.")
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=3600.0,      # allow up to 60 minutes
    prefer_grpc=False
)
collection_name = "arxiv_papers"
# Create collection if it doesn't exist
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
class Embedder:
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    def __init__(self):
        pass
    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        #runs a batch for embedding multiple texts
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings

def retry_operation(func, retries=5, delay=60.0, backoff=2.0, jitter=0.25, *args, **kwargs):
    """
    @brief Retries a remote operation (Supabase/Qdrant) with backoff.
    @param func: The function to retry.
    @param retries: Number of retry attempts.
    @param delay: Initial delay between retries in seconds.
    @param backoff: Multiplier for exponential backoff.
    @param jitter: Random jitter to avoid thundering herd.
    @return: The result of the function if successful, or raises the last exception.
    """
    current_delay = delay
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                wait = current_delay + (jitter * np.random.rand())
                print(f"Operation failed: {e}. Retrying in {wait:.2f}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                current_delay *= backoff
            else:
                print("Operation failed after multiple attempts.")
                raise

def insert_author_into_db(supabase: Client, authorName: str) -> int:
    """
    @brief Inserts an author into the database if not already present.
    @param supabase: Supabase client instance.
    @param authorName: Name of the author to insert.
    @return: ID of the inserted or existing author.
    """
    def operation():
        # Check if author already exists
        existing_author = supabase.table("Authors").select("id").eq("name", authorName).execute()
        if existing_author.data:
            return existing_author.data[0]["id"]
        # Insert new author
        response = supabase.table("Authors").insert({"name": authorName}).execute()
        return response.data[0]["id"]
    
    return retry_operation(operation)

def get_or_create_authors_bulk(supabase: Client, names: list[str]) -> dict[str, int]:
    """
    @brief Resolves author IDs for a batch of names via a single select and bulk insert.
    @param supabase: Supabase client instance.
    @param names: List of author names (may contain duplicates/empties).
    @return: Dict mapping author name -> author id.
    """
    unique_names = sorted({n.strip() for n in names if n and n.strip()})
    if not unique_names:
        return {}

    # Fetch existing authors in one call
    existing = retry_operation(lambda: supabase.table("Authors").select("id,name").in_("name", unique_names).execute())
    name_to_id: dict[str, int] = {row["name"]: row["id"] for row in (existing.data or [])}

    # Insert missing authors in bulk
    missing = [n for n in unique_names if n not in name_to_id]
    if missing:
        to_insert = [{"name": n} for n in missing]
        inserted = retry_operation(lambda: supabase.table("Authors").insert(to_insert).execute())
        for row in (inserted.data or []):
            name_to_id[row["name"]] = row["id"]

    return name_to_id

def insert_papers_into_db(supabase: Client, paperData: list[dict], embedding: list[np.ndarray] = None) -> list[int]: 
    """
    @brief Inserts a paper into the database.
    @param supabase: Supabase client instance.
    @param paperData: Dictionary containing paper data.
    @return: ID of the inserted paper.
    """
    # Aggregate all author names (authors list + submitters) to resolve IDs in bulk
    all_author_names: list[str] = []
    for paper in paperData:
        authors_str = paper.get("authors", "")
        all_author_names.extend([author.strip() for author in authors_str.split(",") if author.strip()])
        all_author_names.append(paper.get("submitter", "Anonymous"))

    name_to_id = get_or_create_authors_bulk(supabase, all_author_names)

    papers_to_insert = []
    for paper in paperData:
        authors_str = paper.get("authors", "")
        authors_list = [author.strip() for author in authors_str.split(",") if author.strip()]
        author_ids = [name_to_id.get(author) for author in authors_list if name_to_id.get(author) is not None]
        submitter_id = name_to_id.get(paper.get("submitter", "Anonymous"), name_to_id.get("Anonymous"))

        # Prepare paper data for batch insertion
        paper_data_to_insert = {
            "title": paper.get("title"),
            "submitter": submitter_id,
            "authors": author_ids,
            "journal_ref": paper.get("journal-ref", ""),
            "doi": paper.get("doi", ""),
            "report_number": paper.get("report-no", ""),
            "categories": paper.get("categories", []),
            "paper_license": paper.get("license", ""),
            "abstract": paper.get("abstract", ""),
            "updated_date": paper.get("updated_date"),
            "comments": paper.get("comments", ""),
        }
        papers_to_insert.append(paper_data_to_insert)

    # Batch insert papers
    # Insert entire batch directly without existence checks
    inserted_resp = retry_operation(lambda: supabase.table("Papers").insert(papers_to_insert).execute())
    paper_ids = [row["id"] for row in (inserted_resp.data or [])]
    #print(f"Inserted {len(inserted_map)} new papers, {len(existing_map)} already existed. Total processed: {len(paper_ids)}.")
    #inset paper into qdrant
    if embedding is not None:
        batch_points = []
        for idx, paper in enumerate(paperData):
            payload = {
                "title": paper.get("title"),
                "submitter": paper.get("submitter", "Anonymous"),
                "authors": paper.get("authors", ""),
                "doi": paper.get("doi", ""),
                "abstract": paper.get("abstract", ""),
                "categories": paper.get("categories", []),
                "journal_ref": paper.get("journal-ref", ""),
                "supaIndex": paper_ids[idx]
            }
            batch_points.append(PointStruct(id=paper_ids[idx], vector=embedding[idx].tolist(), payload=payload))
        retry_operation(lambda: qdrant_client.upsert(collection_name="arxiv_papers", points=batch_points))
    return paper_ids

def main():
    #Use this to populate the articles database from the arXiv dataset. _populateArticles.py will never be called from the main server and is inteded to be run manually when the dataset is updated.
   # Download latest version
    print("Downloading latest version of arXiv dataset...")
    kaggleDSPath = kagglehub.dataset_download("Cornell-University/arxiv")
    print("Path to dataset files:", kaggleDSPath)
    fields_to_ignore = {"id", "versions", "authors_parsed"}
    supabase = get_supabase_client()
    # To enable resuming from last processed line, check Supabase for the last inserted paper index
    last_id = retry_operation(lambda: supabase.table("Papers").select("id").order("id", desc=True).limit(1).execute())
    last_id = int(last_id.data[0]["id"]) if last_id.data else 0

    # Count total lines (optional but gives accurate progress)
    with open(kaggleDSPath + "/arxiv-metadata-oai-snapshot.json", "r") as f:
        totalLines = sum(1 for _ in f)

    batch_size = 50  # Define batch size for processing
    papers_batch = []
    embeddings_batch = []
    embedder_instance = Embedder()

    with open(kaggleDSPath + "/arxiv-metadata-oai-snapshot.json", "r") as f:
        """
        # If last_id exists, skip to that line
        for lineNumber, line in enumerate(tqdm(f, total=totalLines, desc="Processing arXiv dataset")):
            if lineNumber < last_id:
                continue

            data = json.loads(line)
            paperData = {k: v for k, v in data.items() if k not in fields_to_ignore}
            papers_batch.append(paperData)

            # Once the batch is full, process and insert
            if len(papers_batch) == batch_size:
                # Calculate embeddings for the batch
                abstracts = [paper.get("abstract", "") for paper in papers_batch]
                embeddings_batch = embedder_instance.embed_text(abstracts)

                # Insert papers into the database
                insert_papers_into_db(supabase, papers_batch, embeddings_batch)

                # Clear the batches
                papers_batch = []
                embeddings_batch = []

        # Insert any remaining papers in the last batch
        if papers_batch:
            abstracts = [paper.get("abstract", "") for paper in papers_batch]
            embeddings_batch = embedder_instance.embed_text(abstracts)
            insert_papers_into_db(supabase, papers_batch, embeddings_batch)
        """
        #index the qdrant text fields (title, authors, abstract) for better search
        print("Creating Qdrant title text index...")
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=qdrant_models.TextIndexParams(
                type=qdrant_models.TextIndexType.TEXT,
                tokenizer=qdrant_models.TokenizerType.WORD,
                min_token_len = 2,
                max_token_len = 10,
                lowercase = True,
                phrase_matching=True,
            ),
        )
        print("Creating Qdrant authors text index...")
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="authors",
            field_schema=qdrant_models.TextIndexParams(
                type=qdrant_models.TextIndexType.TEXT,
                tokenizer=qdrant_models.TokenizerType.WORD,
                min_token_len = 2,
                max_token_len = 10,
                lowercase = True,
                phrase_matching=True,
            ),
        )  
        print("Creating Qdrant abstract text index...")
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="abstract",
            field_schema=qdrant_models.TextIndexParams(
                type=qdrant_models.TextIndexType.TEXT,
                tokenizer=qdrant_models.TokenizerType.WORD,
                min_token_len = 2,
                max_token_len = 10,
                lowercase = True,
                phrase_matching=True,
            ),
        )
        #index the numeric supaIndex field as prinipal integer index
        print("Creating Qdrant supaIndex integer index...")
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="supaIndex",
            field_schema=qdrant_models.IntegerIndexParams(
                type=qdrant_models.IntegerIndexType.INTEGER,
                is_principal=True,
            ),
        )
        """
        #Remove qdrant entries outside of the supabase range
        print("Cleaning up Qdrant entries outside of Supabase range...")
        #first_id = retry_operation(lambda: supabase.table("Papers").select("id").order("id", desc=True).limit(1).execute())
        #first_id = int(first_id.data[0]["id"]) if first_id.data else 0
        #last_id = retry_operation(lambda: supabase.table("Papers").select("id").order("id", desc=False).limit(1).execute())
        #last_id = int(last_id.data[0]["id"]) if last_id.data else 0
        first_id = 12421
        last_id = 350000
        print(f"Supabase paper ID range: {first_id} to {last_id}")
        # Delete Qdrant points with supaIndex less than first_id
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="supaIndex",
                        range=qdrant_models.Range(
                            lt=first_id,
                        )
                    )
                ]
            )
        )
        # Delete Qdrant points with supaIndex greater than last_id
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="supaIndex",
                        range=qdrant_models.Range(
                            gt=last_id,
                        )
                    )
                ]
            )
        )
        """

def recompute_embeddings_for_all_papers():
    print("Recomputing embeddings for all papers in Qdrant...")
    embedder_instance = Embedder()
    batch_size = 100
    start_index = int(12421 + 230*batch_size)  # Adjusted to skip already processed entries
    max_index = 350000
    #iterate over all papers in qdrant in batches, copy the pyload and rewrite the vector with the new embedding.
    offset = 0 #used to retry in case of empty points by increasing the offset by one until we get a non empty batch
    for idx in tqdm(range(start_index, max_index + 1, batch_size), desc="Recomputing embeddings"):
        while True:
            points=qdrant_client.scroll(
                collection_name=collection_name,
                offset=idx+offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=False
            )
            abstracts = []
            ids = []
            for point in points[0]:
                if point is None:
                    continue
                if point.payload is None:
                    continue
                abstracts.append(point.payload.get("abstract", ""))
                ids.append(point.id)
            print(f"Fetched {len(abstracts)} abstracts at offset {idx+offset}")
            if len(abstracts) > 0:
                embeddings = embedder_instance.embed_text(abstracts)
                batch_points = []
                for id, emb in zip(ids, embeddings):
                    batch_points.append(PointStruct(id=id, vector=emb.tolist()))
                print(f"Updating embeddings for {len(batch_points)} points at offset {idx+offset}")
                print(f"Sample updated point ID {batch_points[0].id} with vector norm {np.linalg.norm(np.array(batch_points[0].vector)):.4f}")
                qdrant_client.update_vectors(
                    collection_name=collection_name,
                    points=batch_points
                )
                break
            else:
                print(f"Warning: No points found at offset {idx+offset}, increasing offset and retrying...")
                offset += 1
                if start_index + offset > max_index:
                    print("No more points to process.")
                    return
    

if __name__ == "__main__":
    #main()
    recompute_embeddings_for_all_papers()