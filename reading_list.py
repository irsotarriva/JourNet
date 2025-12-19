from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from connect_sql import get_supabase_client
from loggin import get_current_user

router = APIRouter()

# Get Supabase client
supabase = get_supabase_client()


# -------------------
# Schemas
# -------------------

class ReadingListCreate(BaseModel):
    name: str
    paperIds: List[int] = []


class ReadingListUpdate(BaseModel):
    name: Optional[str] = None


class ReadingListResponse(BaseModel):
    id: int
    userid: str
    name: str
    paperid: Optional[List[int]]
    created_at: datetime


class AddPaperRequest(BaseModel):
    paperId: int


# -------------------
# Routes
# -------------------

@router.post("/", response_model=ReadingListResponse, status_code=status.HTTP_201_CREATED)
def create_reading_list(data: ReadingListCreate, current_user=Depends(get_current_user)):
    """
    Create a new reading list for the current user with a unique name.
    """
    try:
        # Check if user already has a reading list with this name
        existing = (
            supabase.table("ReadingList")
            .select("*")
            .eq("userid", current_user["id"])
            .eq("name", data.name)
            .execute()
        )

        if existing.data:
            raise HTTPException(
                status_code=400,
                detail=f"Reading list with name '{data.name}' already exists"
            )

        # Create reading list
        reading_list_data = {
            "userid": current_user["id"],
            "name": data.name,
            "paperid": data.paperIds if data.paperIds else []
        }

        response = supabase.table("ReadingList").insert(reading_list_data).execute()

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create reading list"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create reading list: {str(e)}"
        )


@router.get("/", response_model=List[ReadingListResponse])
def get_all_my_reading_lists(current_user=Depends(get_current_user)):
    """
    Get all reading lists for the current user.
    """
    try:
        response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("userid", current_user["id"])
            .order("created_at", desc=True)
            .execute()
        )

        return response.data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve reading lists: {str(e)}"
        )


@router.get("/{list_id}", response_model=ReadingListResponse)
def get_reading_list_by_id(list_id: int, current_user=Depends(get_current_user)):
    """
    Get a specific reading list by ID.
    """
    try:
        response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve reading list: {str(e)}"
        )


@router.get("/name/{list_name}", response_model=ReadingListResponse)
def get_reading_list_by_name(list_name: str, current_user=Depends(get_current_user)):
    """
    Get a reading list by its name.
    """
    try:
        response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("userid", current_user["id"])
            .eq("name", list_name)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Reading list '{list_name}' not found"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve reading list: {str(e)}"
        )


@router.get("/{list_id}/papers")
def get_reading_list_papers(list_id: int, current_user=Depends(get_current_user)):
    """
    Get all papers in a specific reading list with full paper details.
    """
    try:
        # Get reading list
        reading_list_response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not reading_list_response.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        reading_list = reading_list_response.data[0]
        paper_ids = reading_list.get("paperid", [])

        if not paper_ids:
            return {
                "reading_list_id": list_id,
                "reading_list_name": reading_list.get("name"),
                "papers": [],
                "count": 0
            }

        # Get paper details
        papers_response = supabase.table("Papers").select("*").in_("id", paper_ids).execute()

        return {
            "reading_list_id": list_id,
            "reading_list_name": reading_list.get("name"),
            "papers": papers_response.data,
            "count": len(papers_response.data)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve reading list papers: {str(e)}"
        )


@router.patch("/{list_id}", response_model=ReadingListResponse)
def update_reading_list(list_id: int, data: ReadingListUpdate, current_user=Depends(get_current_user)):
    """
    Update a reading list's name.
    """
    try:
        # Check if reading list exists and belongs to user
        existing = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        # Check if new name conflicts with another list
        if data.name:
            name_check = (
                supabase.table("ReadingList")
                .select("*")
                .eq("userid", current_user["id"])
                .eq("name", data.name)
                .neq("id", list_id)
                .execute()
            )

            if name_check.data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Another reading list with name '{data.name}' already exists"
                )

        # Update reading list
        update_data = {}
        if data.name is not None:
            update_data["name"] = data.name

        response = (
            supabase.table("ReadingList")
            .update(update_data)
            .eq("id", list_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update reading list"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update reading list: {str(e)}"
        )


@router.post("/{list_id}/papers", status_code=status.HTTP_200_OK)
def add_paper_to_reading_list(list_id: int, data: AddPaperRequest, current_user=Depends(get_current_user)):
    """
    Add a paper to a specific reading list.
    """
    try:
        # Get reading list
        response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        reading_list = response.data[0]
        current_papers = reading_list.get("paperid", []) or []

        # Check if paper already in list
        if data.paperId in current_papers:
            raise HTTPException(
                status_code=400,
                detail="Paper already in reading list"
            )

        # Add paper to list
        updated_papers = current_papers + [data.paperId]

        update_response = (
            supabase.table("ReadingList")
            .update({"paperid": updated_papers})
            .eq("id", list_id)
            .execute()
        )

        if not update_response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to add paper to reading list"
            )

        return {
            "message": "Paper added to reading list successfully",
            "reading_list": update_response.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add paper to reading list: {str(e)}"
        )


@router.delete("/{list_id}/papers/{paper_id}", status_code=status.HTTP_200_OK)
def remove_paper_from_reading_list(list_id: int, paper_id: int, current_user=Depends(get_current_user)):
    """
    Remove a paper from a specific reading list.
    """
    try:
        # Get reading list
        response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        reading_list = response.data[0]
        current_papers = reading_list.get("paperid", []) or []

        # Check if paper is in list
        if paper_id not in current_papers:
            raise HTTPException(
                status_code=404,
                detail="Paper not found in reading list"
            )

        # Remove paper from list
        updated_papers = [pid for pid in current_papers if pid != paper_id]

        update_response = (
            supabase.table("ReadingList")
            .update({"paperid": updated_papers})
            .eq("id", list_id)
            .execute()
        )

        if not update_response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to remove paper from reading list"
            )

        return {
            "message": "Paper removed from reading list successfully",
            "reading_list": update_response.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove paper from reading list: {str(e)}"
        )


@router.delete("/{list_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_reading_list(list_id: int, current_user=Depends(get_current_user)):
    """
    Delete a specific reading list.
    """
    try:
        # Check if reading list exists and belongs to user
        existing = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        # Delete reading list
        supabase.table("ReadingList").delete().eq("id", list_id).execute()

        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete reading list: {str(e)}"
        )


@router.post("/{list_id}/papers/bulk", status_code=status.HTTP_200_OK)
def add_multiple_papers(list_id: int, paper_ids: List[int], current_user=Depends(get_current_user)):
    """
    Add multiple papers to a reading list at once.
    """
    try:
        # Get reading list
        response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        reading_list = response.data[0]
        current_papers = reading_list.get("paperid", []) or []

        # Add only papers that aren't already in the list
        new_papers = [pid for pid in paper_ids if pid not in current_papers]
        updated_papers = current_papers + new_papers

        update_response = (
            supabase.table("ReadingList")
            .update({"paperid": updated_papers})
            .eq("id", list_id)
            .execute()
        )

        if not update_response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to add papers to reading list"
            )

        return {
            "message": f"Added {len(new_papers)} new papers to reading list",
            "added_count": len(new_papers),
            "skipped_count": len(paper_ids) - len(new_papers),
            "reading_list": update_response.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add papers to reading list: {str(e)}"
        )


@router.delete("/{list_id}/papers", status_code=status.HTTP_200_OK)
def clear_reading_list(list_id: int, current_user=Depends(get_current_user)):
    """
    Clear all papers from a reading list (but keep the list itself).
    """
    try:
        # Check if reading list exists and belongs to user
        response = (
            supabase.table("ReadingList")
            .select("*")
            .eq("id", list_id)
            .eq("userid", current_user["id"])
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Reading list not found"
            )

        # Clear papers
        update_response = (
            supabase.table("ReadingList")
            .update({"paperid": []})
            .eq("id", list_id)
            .execute()
        )

        if not update_response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear reading list"
            )

        return {
            "message": "Reading list cleared successfully",
            "reading_list": update_response.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear reading list: {str(e)}"
        )