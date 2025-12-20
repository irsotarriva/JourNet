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

class PaperCreate(BaseModel):
    title: str
    submitter: Optional[int] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    report_number: Optional[str] = None
    categories: Optional[str] = None
    paper_license: Optional[str] = None
    abstract: Optional[str] = None
    updated_date: Optional[datetime] = None
    comments: Optional[str] = None
    authors: Optional[List[int]] = None


class PaperUpdate(BaseModel):
    title: Optional[str] = None
    submitter: Optional[int] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    report_number: Optional[str] = None
    categories: Optional[str] = None
    paper_license: Optional[str] = None
    abstract: Optional[str] = None
    updated_date: Optional[datetime] = None
    comments: Optional[str] = None
    authors: Optional[List[int]] = None
    comments_summary: Optional[str] = None


class PaperResponse(BaseModel):
    id: int
    title: Optional[str]
    submitter: Optional[int]
    journal_ref: Optional[str]
    doi: Optional[str]
    report_number: Optional[str]
    categories: Optional[str]
    paper_license: Optional[str]
    abstract: Optional[str]
    updated_date: Optional[datetime]
    comments: Optional[str]
    authors: Optional[List[int]]
    comments_summary: Optional[str]


# -------------------
# Routes
# -------------------

@router.post("/", response_model=PaperResponse, status_code=status.HTTP_201_CREATED)
def create_paper(data: PaperCreate, current_user=Depends(get_current_user)):
    """
    Create a new paper in the Papers table.
    Requires authentication.
    """
    try:
<<<<<<< HEAD
        return create_paper_db(data)
=======
        paper_data = {
            "title": data.title,
            "submitter": data.submitter,
            "journal_ref": data.journal_ref,
            "doi": data.doi,
            "report_number": data.report_number,
            "categories": data.categories,
            "paper_license": data.paper_license,
            "abstract": data.abstract,
            "updated_date": data.updated_date.isoformat() if data.updated_date else None,
            "comments": data.comments,
            "authors": data.authors
        }

        response = supabase.table("Papers").insert(paper_data).execute()

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create paper"
            )

        return response.data[0]

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create paper: {str(e)}"
        )


@router.get("/{paper_id}", response_model=PaperResponse)
def get_paper(paper_id: int):
    """
    Get a specific paper by ID.
    """
    try:
<<<<<<< HEAD
        return get_paper_by_id_db(paper_id)
=======
        response = supabase.table("Papers").select("*").eq("id", paper_id).execute()

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Paper not found"
            )

        return response.data[0]

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve paper: {str(e)}"
        )


@router.get("/")
def get_all_papers(limit: int = 50, offset: int = 0):
    """
    Get all papers with pagination.
    """
    try:
<<<<<<< HEAD
        return get_all_papers_db(limit=limit, offset=offset)
=======
        response = (
            supabase.table("Papers")
            .select("*")
            .order("updated_date", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return {
            "papers": response.data,
            "count": len(response.data),
            "offset": offset,
            "limit": limit
        }

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve papers: {str(e)}"
        )


@router.patch("/{paper_id}", response_model=PaperResponse)
def update_paper(paper_id: int, data: PaperUpdate, current_user=Depends(get_current_user)):
    """
    Update a paper.
    Requires authentication.
    """
    try:
<<<<<<< HEAD
        return update_paper_db(paper_id, data)
=======
        # Check if paper exists
        existing = supabase.table("Papers").select("*").eq("id", paper_id).execute()

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Paper not found"
            )

        # Prepare update data
        update_data = {}
        if data.title is not None:
            update_data["title"] = data.title
        if data.submitter is not None:
            update_data["submitter"] = data.submitter
        if data.journal_ref is not None:
            update_data["journal_ref"] = data.journal_ref
        if data.doi is not None:
            update_data["doi"] = data.doi
        if data.report_number is not None:
            update_data["report_number"] = data.report_number
        if data.categories is not None:
            update_data["categories"] = data.categories
        if data.paper_license is not None:
            update_data["paper_license"] = data.paper_license
        if data.abstract is not None:
            update_data["abstract"] = data.abstract
        if data.updated_date is not None:
            update_data["updated_date"] = data.updated_date.isoformat()
        if data.comments is not None:
            update_data["comments"] = data.comments
        if data.authors is not None:
            update_data["authors"] = data.authors
        if data.comments_summary is not None:
            update_data["comments_summary"] = data.comments_summary

        response = (
            supabase.table("Papers")
            .update(update_data)
            .eq("id", paper_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update paper"
            )

        return response.data[0]

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update paper: {str(e)}"
        )


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_paper(paper_id: int, current_user=Depends(get_current_user)):
    """
    Delete a paper.
    Requires authentication.
    """
    try:
<<<<<<< HEAD
        delete_paper_db(paper_id)
        return None
=======
        # Check if paper exists
        existing = supabase.table("Papers").select("*").eq("id", paper_id).execute()

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Paper not found"
            )

        # Delete paper
        supabase.table("Papers").delete().eq("id", paper_id).execute()

        return None

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete paper: {str(e)}"
        )


@router.get("/search/by-title")
def search_papers_by_title(query: str, limit: int = 20):
    """
    Search papers by title.
    """
    try:
<<<<<<< HEAD
        return search_papers_by_title_db(query, limit)
=======
        response = (
            supabase.table("Papers")
            .select("*")
            .ilike("title", f"%{query}%")
            .limit(limit)
            .execute()
        )

        return {
            "papers": response.data,
            "count": len(response.data),
            "query": query
        }

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search papers: {str(e)}"
        )


@router.get("/search/by-category")
def search_papers_by_category(category: str, limit: int = 50):
    """
    Search papers by category.
    """
    try:
<<<<<<< HEAD
        return search_papers_by_category_db(category, limit)
=======
        response = (
            supabase.table("Papers")
            .select("*")
            .ilike("categories", f"%{category}%")
            .limit(limit)
            .execute()
        )

        return {
            "papers": response.data,
            "count": len(response.data),
            "category": category
        }

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search papers: {str(e)}"
        )


@router.get("/{paper_id}/abstract")
def get_paper_abstract(paper_id: int):
    """
    Get only the abstract of a paper.
    """
    try:
<<<<<<< HEAD
        return get_paper_abstract_db(paper_id)
=======
        response = (
            supabase.table("Papers")
            .select("id, title, abstract")
            .eq("id", paper_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Paper not found"
            )

        return response.data[0]

>>>>>>> 8d0535e (interact with papers database with papers.py)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve abstract: {str(e)}"
<<<<<<< HEAD
        )


# -------------------
# Database Helper Functions
# -------------------

def create_paper_db(data: PaperCreate):
    paper_data = {
        "title": data.title,
        "submitter": data.submitter,
        "journal_ref": data.journal_ref,
        "doi": data.doi,
        "report_number": data.report_number,
        "categories": data.categories,
        "paper_license": data.paper_license,
        "abstract": data.abstract,
        "updated_date": data.updated_date.isoformat() if data.updated_date else None,
        "comments": data.comments,
        "authors": data.authors
    }

    response = supabase.table("Papers").insert(paper_data).execute()

    if not response.data:
        raise Exception("Failed to create paper")

    return response.data[0]


def get_paper_by_id_db(paper_id: int):
    response = supabase.table("Papers").select("*").eq("id", paper_id).execute()

    if not response.data:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    return response.data[0]


def get_all_papers_db(limit: int = 50, offset: int = 0):
    response = (
        supabase.table("Papers")
        .select("*")
        .order("updated_date", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    return {
        "papers": response.data,
        "count": len(response.data),
        "offset": offset,
        "limit": limit
    }


def update_paper_db(paper_id: int, data: PaperUpdate):
    # Check if paper exists
    existing = supabase.table("Papers").select("*").eq("id", paper_id).execute()

    if not existing.data:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    # Prepare update data
    update_data = {}
    if data.title is not None:
        update_data["title"] = data.title
    if data.submitter is not None:
        update_data["submitter"] = data.submitter
    if data.journal_ref is not None:
        update_data["journal_ref"] = data.journal_ref
    if data.doi is not None:
        update_data["doi"] = data.doi
    if data.report_number is not None:
        update_data["report_number"] = data.report_number
    if data.categories is not None:
        update_data["categories"] = data.categories
    if data.paper_license is not None:
        update_data["paper_license"] = data.paper_license
    if data.abstract is not None:
        update_data["abstract"] = data.abstract
    if data.updated_date is not None:
        update_data["updated_date"] = data.updated_date.isoformat()
    if data.comments is not None:
        update_data["comments"] = data.comments
    if data.authors is not None:
        update_data["authors"] = data.authors
    if data.comments_summary is not None:
        update_data["comments_summary"] = data.comments_summary

    response = (
        supabase.table("Papers")
        .update(update_data)
        .eq("id", paper_id)
        .execute()
    )

    if not response.data:
        raise Exception("Failed to update paper")

    return response.data[0]


def delete_paper_db(paper_id: int):
    # Check if paper exists
    existing = supabase.table("Papers").select("*").eq("id", paper_id).execute()

    if not existing.data:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    # Delete paper
    supabase.table("Papers").delete().eq("id", paper_id).execute()


def search_papers_by_title_db(query: str, limit: int = 20):
    response = (
        supabase.table("Papers")
        .select("*")
        .ilike("title", f"%{query}%")
        .limit(limit)
        .execute()
    )

    return {
        "papers": response.data,
        "count": len(response.data),
        "query": query
    }


def search_papers_by_category_db(category: str, limit: int = 50):
    response = (
        supabase.table("Papers")
        .select("*")
        .ilike("categories", f"%{category}%")
        .limit(limit)
        .execute()
    )

    return {
        "papers": response.data,
        "count": len(response.data),
        "category": category
    }


def get_paper_abstract_db(paper_id: int):
    response = (
        supabase.table("Papers")
        .select("id, title, abstract")
        .eq("id", paper_id)
        .execute()
    )

    if not response.data:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    return response.data[0]
=======
        )
>>>>>>> 8d0535e (interact with papers database with papers.py)
