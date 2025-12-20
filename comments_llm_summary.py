from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import requests
from connect_sql import get_supabase_client
from loggin import get_current_user
<<<<<<< HEAD
from papers import get_paper_by_id_db
=======
>>>>>>> 8d0535e (interact with papers database with papers.py)

router = APIRouter()

# Load environment variables
load_dotenv()

# Get Supabase client
supabase = get_supabase_client()

# Hugging Face API configuration
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
if not HUGGING_FACE_API_KEY:
    print("WARNING: HUGGING_FACE_API_KEY not found in .env file")

<<<<<<< HEAD
HUGGING_FACE_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
=======
HUGGING_FACE_API_URL = "https://router.huggingface.co/models/facebook/bart-large-cnn"
>>>>>>> 8d0535e (interact with papers database with papers.py)


# -------------------
# Schemas
# -------------------

class SummaryResponse(BaseModel):
    paper_id: int
    comments_count: int
    summary: Optional[str]
    updated: bool
    message: str


# -------------------
# Helper Functions
# -------------------

def get_all_comments_for_paper(paper_id: int):
    """
    Retrieve all comments for a paper from the Discussion table.
    Includes upVotes and downVotes for scoring.
    """
    try:
        response = (
            supabase.table("Discussion")
            .select("id, comment, isAnonymous, userId, created_at, upVotes, downVotes")
            .eq("articleId", paper_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data
    except Exception as e:
        raise Exception(f"Failed to retrieve comments: {str(e)}")


def calculate_comment_score(comment):
    """
    Calculate the score of a comment based on upvotes and downvotes.
    Score = upVotes - downVotes
    """
    up_votes = comment.get("upVotes", 0) or 0
    down_votes = comment.get("downVotes", 0) or 0
    return up_votes - down_votes


<<<<<<< HEAD
def prepare_weighted_comments_text(comments, paper_abstract):
    """
    Prepare text for summarization with weighted comments and paper abstract.
    - Filters out comments with negative scores (score < 0)
=======
def get_paper_abstract(paper_id: int):
    """
    Retrieve the abstract of a paper from the Papers table.
    """
    try:
        response = (
            supabase.table("Papers")
            .select("abstract")
            .eq("id", paper_id)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0].get("abstract", "")
        return ""
    except Exception as e:
        raise Exception(f"Failed to retrieve paper abstract: {str(e)}")


def prepare_weighted_comments_text(comments, paper_abstract):
    """
    Prepare text for summarization with weighted comments and paper abstract.
    - Filters out comments with negative scores
>>>>>>> 8d0535e (interact with papers database with papers.py)
    - Weights comments by positive scores (higher score = more repetitions)
    - Adds paper abstract as context
    """
    # Filter and score comments
    scored_comments = []
    for comment in comments:
        comment_text = comment.get("comment", "")
        if not comment_text:
            continue

        score = calculate_comment_score(comment)

<<<<<<< HEAD
        # Only include comments with non-negative scores (>= 0)
        if score >= 0:
=======
        # Only include comments with positive scores
        if score > 0:
>>>>>>> 8d0535e (interact with papers database with papers.py)
            scored_comments.append({
                "text": comment_text,
                "score": score
            })

    # Sort by score (highest first)
    scored_comments.sort(key=lambda x: x["score"], reverse=True)

    # Build the text with paper abstract first
    text_parts = []

    # Add paper abstract as context if available
    if paper_abstract:
        text_parts.append(f"Paper Abstract: {paper_abstract}")
        text_parts.append("")  # Empty line for separation

    # Add weighted comments
<<<<<<< HEAD
=======
    # For simplicity, we'll repeat high-scored comments to give them more weight
    # Comments with score >= 5 get repeated 3 times
    # Comments with score 3-4 get repeated 2 times
    # Comments with score 1-2 get included once
>>>>>>> 8d0535e (interact with papers database with papers.py)
    text_parts.append("Discussion Comments:")
    for i, comment_data in enumerate(scored_comments, 1):
        score = comment_data["score"]
        text = comment_data["text"]

<<<<<<< HEAD
        if score >= 10:
=======
        if score >= 5:
>>>>>>> 8d0535e (interact with papers database with papers.py)
            # High importance - repeat 3 times
            text_parts.append(f"[High-rated comment {i}]: {text}")
            text_parts.append(f"[Important]: {text}")
            text_parts.append(f"[Highly upvoted]: {text}")
<<<<<<< HEAD
        elif score >= 5:
=======
        elif score >= 3:
>>>>>>> 8d0535e (interact with papers database with papers.py)
            # Medium importance - repeat 2 times
            text_parts.append(f"[Well-received comment {i}]: {text}")
            text_parts.append(f"[Upvoted]: {text}")
        else:
<<<<<<< HEAD
            # Low importance (score 0-2) - include once
=======
            # Low importance - include once
>>>>>>> 8d0535e (interact with papers database with papers.py)
            text_parts.append(f"Comment {i}: {text}")

    return " ".join(text_parts)


def generate_summary_with_huggingface(text: str) -> str:
    """
    Generate a summary using Hugging Face's BART model.
    """
    if not HUGGING_FACE_API_KEY:
        raise Exception("Hugging Face API key not configured")

    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
    }

    # Prepare payload
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 200,
            "min_length": 50,
            "do_sample": False
        }
    }

    try:
        response = requests.post(
            HUGGING_FACE_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
<<<<<<< HEAD
            try:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("summary_text", "")
                else:
                    raise Exception(f"Unexpected API response format: {result}")
            except requests.exceptions.JSONDecodeError:
                raise Exception(f"Failed to parse successful response: {response.text[:200]}")
        else:
            try:
                error_msg = response.json().get("error", "Unknown error")
            except requests.exceptions.JSONDecodeError:
                error_msg = response.text[:200]
            
            raise Exception(f"Hugging Face API error (Status {response.status_code}): {error_msg}")
=======
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", "")
            else:
                raise Exception("Unexpected API response format")
        else:
            error_msg = response.json().get("error", "Unknown error")
            raise Exception(f"Hugging Face API error: {error_msg}")
>>>>>>> 8d0535e (interact with papers database with papers.py)

    except requests.exceptions.Timeout:
        raise Exception("Hugging Face API request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request error: {str(e)}")


def update_paper_summary(paper_id: int, summary: str):
    """
    Update the comments_summary column in the Papers table.
    """
    try:
        response = (
            supabase.table("Papers")
            .update({"comments_summary": summary})
            .eq("id", paper_id)
            .execute()
        )

        if not response.data:
            raise Exception("Failed to update paper summary")

        return response.data[0]
    except Exception as e:
        raise Exception(f"Failed to update paper: {str(e)}")


# -------------------
# Routes
# -------------------

@router.post("/{paper_id}/summarize", response_model=SummaryResponse)
def summarize_paper_comments(paper_id: int, current_user=Depends(get_current_user)):
    """
    Generate and save a summary of comments for a paper.
    Only generates summary if paper has more than 10 comments.
    Requires authentication.
    """
    try:
<<<<<<< HEAD
        # Check if paper exists and get abstract
        try:
            paper_data = get_paper_by_id_db(paper_id)
            paper_abstract = paper_data.get("abstract", "")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve paper: {str(e)}"
=======
        # Check if paper exists
        paper_response = supabase.table("Papers").select("id, title").eq("id", paper_id).execute()
        if not paper_response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Paper with ID {paper_id} not found"
>>>>>>> 8d0535e (interact with papers database with papers.py)
            )

        # Get all comments for the paper
        comments = get_all_comments_for_paper(paper_id)
        comments_count = len(comments)

        # Check if there are enough comments
        if comments_count < 10:
            return SummaryResponse(
                paper_id=paper_id,
                comments_count=comments_count,
                summary=None,
                updated=False,
                message=f"Paper has only {comments_count} comments. Need at least 10 comments to generate summary."
            )

<<<<<<< HEAD
        # Prepare text for summarization (with abstract and scoring logic)
        full_text = prepare_weighted_comments_text(comments, paper_abstract)
=======
        # Prepare text for summarization
        # Concatenate all comments with separators
        comment_texts = []
        for i, comment in enumerate(comments, 1):
            comment_text = comment.get("comment", "")
            if comment_text:
                # Add comment number for context
                comment_texts.append(f"Comment {i}: {comment_text}")

        full_text = " ".join(comment_texts)
>>>>>>> 8d0535e (interact with papers database with papers.py)

        # Check if text is too long (BART has a limit)
        # If too long, truncate intelligently
        MAX_CHARS = 5000
        if len(full_text) > MAX_CHARS:
            full_text = full_text[:MAX_CHARS] + "..."

        # Generate summary using Hugging Face
        try:
            summary = generate_summary_with_huggingface(full_text)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate summary: {str(e)}"
            )

        if not summary:
            raise HTTPException(
                status_code=500,
                detail="Summary generation returned empty result"
            )

        # Update the paper with the summary
        try:
            update_paper_summary(paper_id, summary)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save summary to database: {str(e)}"
            )

        return SummaryResponse(
            paper_id=paper_id,
            comments_count=comments_count,
            summary=summary,
            updated=True,
            message=f"Successfully generated and saved summary from {comments_count} comments."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to summarize comments: {str(e)}"
        )


@router.get("/{paper_id}/summary")
def get_paper_comments_summary(paper_id: int):
    """
    Get the existing comments summary for a paper.
    Does not generate a new summary - use POST to generate.
    """
    try:
        response = (
            supabase.table("Papers")
            .select("id, title, comments_summary")
            .eq("id", paper_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Paper not found"
            )

        paper = response.data[0]

        return {
            "paper_id": paper["id"],
            "title": paper.get("title"),
            "comments_summary": paper.get("comments_summary"),
            "has_summary": bool(paper.get("comments_summary"))
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve summary: {str(e)}"
        )


@router.post("/batch-summarize")
def batch_summarize_papers(paper_ids: list[int], current_user=Depends(get_current_user)):
    """
    Generate summaries for multiple papers at once.
    Only processes papers with 10+ comments.
    Returns results for each paper.
    """
    results = []

    for paper_id in paper_ids:
        try:
<<<<<<< HEAD
            # Check if paper exists and get abstract
            try:
                paper_data = get_paper_by_id_db(paper_id)
                paper_abstract = paper_data.get("abstract", "")
            except:
                results.append({
                    "paper_id": paper_id,
                    "success": False,
                    "message": "Paper not found",
                    "comments_count": 0
                })
                continue

=======
>>>>>>> 8d0535e (interact with papers database with papers.py)
            # Get comments count
            comments = get_all_comments_for_paper(paper_id)
            comments_count = len(comments)

            if comments_count < 10:
                results.append({
                    "paper_id": paper_id,
                    "success": False,
                    "message": f"Only {comments_count} comments (need 10+)",
                    "comments_count": comments_count
                })
                continue

<<<<<<< HEAD
            # Prepare text for summarization
            full_text = prepare_weighted_comments_text(comments, paper_abstract)
=======
            # Prepare and summarize
            comment_texts = [f"Comment {i}: {c.get('comment', '')}"
                           for i, c in enumerate(comments, 1) if c.get('comment')]
            full_text = " ".join(comment_texts)
>>>>>>> 8d0535e (interact with papers database with papers.py)

            # Truncate if needed
            MAX_CHARS = 5000
            if len(full_text) > MAX_CHARS:
                full_text = full_text[:MAX_CHARS] + "..."

            # Generate summary
            summary = generate_summary_with_huggingface(full_text)

            # Update paper
            update_paper_summary(paper_id, summary)

            results.append({
                "paper_id": paper_id,
                "success": True,
                "message": "Summary generated successfully",
                "comments_count": comments_count,
                "summary": summary
            })

        except Exception as e:
            results.append({
                "paper_id": paper_id,
                "success": False,
                "message": str(e),
                "comments_count": 0
            })

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    return {
        "total_papers": len(paper_ids),
        "successful": successful,
        "failed": failed,
        "results": results
    }


@router.delete("/{paper_id}/summary")
def delete_paper_comments_summary(paper_id: int, current_user=Depends(get_current_user)):
    """
    Delete the comments summary for a paper.
    Sets comments_summary to NULL.
    """
    try:
        # Check if paper exists
        paper_response = supabase.table("Papers").select("id").eq("id", paper_id).execute()
        if not paper_response.data:
            raise HTTPException(
                status_code=404,
                detail="Paper not found"
            )

        # Delete summary
        response = (
            supabase.table("Papers")
            .update({"comments_summary": None})
            .eq("id", paper_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete summary"
            )

        return {
            "message": "Summary deleted successfully",
            "paper_id": paper_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete summary: {str(e)}"
        )