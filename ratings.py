from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from connect_sql import get_supabase_client
from loggin import get_current_user

router = APIRouter()


# -------------------
# Schemas
# -------------------

class RatingCreate(BaseModel):
    paperid: int
    rating: int = Field(..., ge=1, le=5, description="Rating must be between 1 and 5")


class RatingUpdate(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating must be between 1 and 5")


class RatingResponse(BaseModel):
    id: int
    paperid: int
    userid: str
    rating: int
    created_at: datetime


# -------------------
# Routes
# -------------------

@router.post("/", response_model=RatingResponse, status_code=status.HTTP_201_CREATED)
def create_rating(data: RatingCreate, current_user=Depends(get_current_user)):
    """
    Add a rating for a paper.
    Users can only have one rating per paper - use PUT to update.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        # Check if user already rated this paper
        existing = (
            supabase.table("Ratings")
            .select("*")
            .eq("userid", current_user["id"])
            .eq("paperid", data.paperid)
            .execute()
        )

        if existing.data:
            raise HTTPException(
                status_code=400,
                detail=f"You already rated this paper. Use PUT /ratings/{existing.data[0]['id']} to update your rating."
            )

        # Create rating
        rating_data = {
            "userid": current_user["id"],
            "paperid": data.paperid,
            "rating": data.rating
        }

        response = supabase.table("Ratings").insert(rating_data).execute()

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create rating"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for foreign key constraint violation
        if "foreign key constraint" in error_msg.lower() and "paperid" in error_msg:
            raise HTTPException(
                status_code=400,
                detail=f"Paper with ID {data.paperid} does not exist."
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create rating: {str(e)}"
        )


@router.get("/paper/{paper_id}/average")
def get_paper_average_rating(paper_id: int):
    """
    Get the average rating for a paper.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        response = (
            supabase.table("Ratings")
            .select("rating")
            .eq("paperid", paper_id)
            .execute()
        )

        if not response.data:
            return {
                "paper_id": paper_id,
                "average_rating": None,
                "total_ratings": 0
            }

        ratings = [r["rating"] for r in response.data]
        average = sum(ratings) / len(ratings)

        return {
            "paper_id": paper_id,
            "average_rating": round(average, 2),
            "total_ratings": len(ratings)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate average rating: {str(e)}"
        )


@router.get("/paper/{paper_id}/distribution")
def get_paper_rating_distribution(paper_id: int):
    """
    Get the distribution of ratings (1-5 stars) for a paper.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        response = (
            supabase.table("Ratings")
            .select("rating")
            .eq("paperid", paper_id)
            .execute()
        )

        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for rating_obj in response.data:
            rating = rating_obj["rating"]
            if 1 <= rating <= 5:
                distribution[rating] += 1

        total = sum(distribution.values())

        return {
            "paper_id": paper_id,
            "distribution": distribution,
            "total_ratings": total,
            "average_rating": round(sum(k * v for k, v in distribution.items()) / total, 2) if total > 0 else None
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rating distribution: {str(e)}"
        )


@router.get("/paper/{paper_id}/my-rating")
def get_my_rating_for_paper(paper_id: int, current_user=Depends(get_current_user)):
    """
    Get the current user's rating for a specific paper.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        response = (
            supabase.table("Ratings")
            .select("*")
            .eq("userid", current_user["id"])
            .eq("paperid", paper_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="You haven't rated this paper yet"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve rating: {str(e)}"
        )


@router.get("/my-ratings")
def get_all_my_ratings(current_user=Depends(get_current_user)):
    """
    Get all ratings by the current user.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        response = (
            supabase.table("Ratings")
            .select("*")
            .eq("userid", current_user["id"])
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "ratings": response.data,
            "count": len(response.data)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve ratings: {str(e)}"
        )


@router.get("/{rating_id}", response_model=RatingResponse)
def get_rating_by_id(rating_id: int, current_user=Depends(get_current_user)):
    """
    Get a specific rating by ID.
    Only returns if the rating belongs to the current user.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        response = (
            supabase.table("Ratings")
            .select("*")
            .eq("id", rating_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Rating not found"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve rating: {str(e)}"
        )


@router.put("/{rating_id}", response_model=RatingResponse)
def update_rating(rating_id: int, data: RatingUpdate, current_user=Depends(get_current_user)):
    """
    Update an existing rating.
    Users can only update their own ratings.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        # Check if rating exists and belongs to user
        existing = (
            supabase.table("Ratings")
            .select("*")
            .eq("id", rating_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Rating not found or you don't have permission to update it"
            )

        # Update rating
        response = (
            supabase.table("Ratings")
            .update({"rating": data.rating})
            .eq("id", rating_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update rating"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update rating: {str(e)}"
        )


@router.delete("/{rating_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rating(rating_id: int, current_user=Depends(get_current_user)):
    """
    Delete a rating.
    Users can only delete their own ratings.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        # Check if rating exists and belongs to user
        existing = (
            supabase.table("Ratings")
            .select("*")
            .eq("id", rating_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Rating not found or you don't have permission to delete it"
            )

        # Delete rating
        supabase.table("Ratings").delete().eq("id", rating_id).execute()

        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete rating: {str(e)}"
        )


@router.get("/paper/{paper_id}")
def get_all_ratings_for_paper(paper_id: int, limit: int = 50, offset: int = 0):
    """
    Get all ratings for a specific paper (paginated).
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        response = (
            supabase.table("Ratings")
            .select("*")
            .eq("paperid", paper_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return {
            "paper_id": paper_id,
            "ratings": response.data,
            "count": len(response.data),
            "offset": offset,
            "limit": limit
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve ratings: {str(e)}"
        )


@router.get("/top-rated")
def get_top_rated_papers(limit: int = 10, min_ratings: int = 5):
    """
    Get top-rated papers based on average rating.
    Only includes papers with at least min_ratings ratings.
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        # Get all ratings
        response = supabase.table("Ratings").select("paperid, rating").execute()

        if not response.data:
            return {
                "papers": [],
                "count": 0
            }

        # Calculate average ratings per paper
        paper_ratings = {}
        for rating in response.data:
            paper_id = rating["paperid"]
            if paper_id not in paper_ratings:
                paper_ratings[paper_id] = []
            paper_ratings[paper_id].append(rating["rating"])

        # Filter papers with minimum ratings and calculate averages
        top_papers = []
        for paper_id, ratings in paper_ratings.items():
            if len(ratings) >= min_ratings:
                avg_rating = sum(ratings) / len(ratings)
                top_papers.append({
                    "paper_id": paper_id,
                    "average_rating": round(avg_rating, 2),
                    "total_ratings": len(ratings)
                })

        # Sort by average rating
        top_papers.sort(key=lambda x: x["average_rating"], reverse=True)

        return {
            "papers": top_papers[:limit],
            "count": len(top_papers[:limit]),
            "criteria": {
                "min_ratings": min_ratings,
                "limit": limit
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get top-rated papers: {str(e)}"
        )