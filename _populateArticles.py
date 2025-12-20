import kagglehub
import numpy as np
import json
import time  # Added for retry mechanism
from tqdm.auto import tqdm
from connect_sql import get_supabase_client
from supabase import Client

def retry_operation(func, retries=3, delay=5, *args, **kwargs):
    """
    @brief Retries a database operation in case of failure.
    @param func: The function to retry.
    @param retries: Number of retry attempts.
    @param delay: Delay between retries in seconds.
    @return: The result of the function if successful, or raises the last exception.
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"Operation failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
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

def insert_paper_into_db(supabase: Client, paperData: dict) -> int:
    """
    @brief Inserts a paper into the database.
    @param supabase: Supabase client instance.
    @param paperData: Dictionary containing paper data.
    @return: ID of the inserted paper.
    """
    authors_str = paperData.pop("authors", "")
    authors_list = [author.strip() for author in authors_str.split(",") if author.strip()]
    author_ids = [insert_author_into_db(supabase, author) for author in authors_list]
    submitter_id = insert_author_into_db(supabase, paperData.get("submitter", "Anonymous"))

    paper_data_to_insert = {
        "title": paperData.get("title"),
        "submitter": submitter_id,
        "authors": author_ids,
        "journal_ref": paperData.get("journal-ref", ""),
        "doi": paperData.get("doi", ""),
        "report_number": paperData.get("report-no", ""),
        "categories": paperData.get("categories", []),
        "paper_license": paperData.get("license", ""),
        "abstract": paperData.get("abstract", ""),
        "updated_date": paperData.get("updated_date"),
        "comments": paperData.get("comments", ""),
    }

    def operation():
        response = supabase.table("Papers").insert(paper_data_to_insert).execute()
        return response.data[0]["id"]

    return retry_operation(operation)

def main():
    # Download latest version
    print("Downloading latest version of arXiv dataset...")
    kaggleDSPath = kagglehub.dataset_download("Cornell-University/arxiv")
    print("Path to dataset files:", kaggleDSPath)
    fields_to_ignore = {"id", "versions", "authors_parsed"}
    supabase = get_supabase_client()
    # To enable resuming from last processed line check supabase for last inserted paper index
    last_id = retry_operation(lambda: supabase.table("Papers").select("id").order("id", desc=True).limit(1).execute())
    last_id = int(last_id.data[0]["id"]) if last_id.data else 0
    # Count total lines (optional but gives accurate progress)
    with open(kaggleDSPath + "/arxiv-metadata-oai-snapshot.json", "r") as f:
        totalLines = sum(1 for _ in f)
    with open(kaggleDSPath + "/arxiv-metadata-oai-snapshot.json", "r") as f:
        # If last_id exists, skip to that line
        for lineNumber, line in enumerate(tqdm(f, total=totalLines, desc="Processing arXiv dataset")):
            if lineNumber < last_id:
                continue
            data = json.loads(line)
            paperData = {k: v for k, v in data.items() if k not in fields_to_ignore}
            insert_paper_into_db(supabase, paperData)


if __name__ == "__main__":
    main()