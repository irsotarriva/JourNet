from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from connect_sql import get_supabase_client
from loggin import get_current_user

router = APIRouter()

# Get Supabase client
supabase = get_supabase_client()


# -------------------
# Schemas
# -------------------

class CommentCreate(BaseModel):
    parent: Optional[int] = None
    isAnonymous: bool = False
    onReview: bool = False
    comment: str
    articleId: int


class CommentResponse(BaseModel):
    id: int
    parent: Optional[int] 
    userId: str
    isAnonymous: Optional[bool] = False
    onReview: Optional[bool] = False
    comment: str
    upVotes: Optional[int] = 0
    downVotes: Optional[int] = 0
    articleId: int
    created_at: datetime
    last_updated: Optional[datetime]


class CommentUpdate(BaseModel):
    comment: Optional[str] = None
    isAnonymous: Optional[bool] = None
    onReview: Optional[bool] = None


# -------------------
# Routes
# -------------------

@router.post("/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
def create_comment(data: CommentCreate, current_user=Depends(get_current_user)):
    """
    Create a new comment in the Discussion table.
    Requires authentication.
    """
    try:
        # Prepare comment data
        # For top-level comments, don't include parent field at all
        comment_data = {
            "userId": current_user["id"],
            "isAnonymous": data.isAnonymous,
            "onReview": data.onReview,
            "comment": data.comment,
            "upVotes": 0,
            "downVotes": 0,
            "articleId": data.articleId,
            "last_updated": datetime.utcnow().isoformat()
        }

        # Only add parent if it's a reply to another comment
        if data.parent is not None and data.parent > 0:
            comment_data["parent"] = data.parent

        # Insert into Discussion table
        response = supabase.table("Discussion").insert(comment_data).execute()

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create comment"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        # Check for foreign key constraint violation
        error_msg = str(e)
        if "foreign key constraint" in error_msg.lower() and "articleId" in error_msg:
            raise HTTPException(
                status_code=400,
                detail=f"Article with ID {data.articleId} does not exist. Please provide a valid articleId."
            )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to create comment: {str(e)}"
        )


@router.get("/{comment_id}", response_model=CommentResponse)
def get_comment(comment_id: int):
    """
    Get a specific comment by ID.
    """
    try:
        response = supabase.table("Discussion").select("*").eq("id", comment_id).execute()

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Comment not found"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve comment: {str(e)}"
        )


@router.get("/article/{article_id}")
def get_article_comments(article_id: int, limit: int = 50, offset: int = 0):
    """
    Get all comments for a specific article.
    Supports pagination with limit and offset.
    """
    try:
        response = (
            supabase.table("Discussion")
            .select("*")
            .eq("articleId", article_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return {
            "comments": response.data,
            "count": len(response.data),
            "offset": offset,
            "limit": limit
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve comments: {str(e)}"
        )


@router.patch("/{comment_id}", response_model=CommentResponse)
def update_comment(comment_id: int, data: CommentUpdate, current_user=Depends(get_current_user)):
    """
    Update an existing comment.
    Only the comment author can update their comment.
    """
    try:
        # First, check if comment exists and belongs to user
        existing = supabase.table("Discussion").select("*").eq("id", comment_id).execute()

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Comment not found"
            )

        if existing.data[0]["userId"] != current_user["id"]:
            raise HTTPException(
                status_code=403,
                detail="You can only update your own comments"
            )

        # Prepare update data
        update_data = {}
        if data.comment is not None:
            update_data["comment"] = data.comment
        if data.isAnonymous is not None:
            update_data["isAnonymous"] = data.isAnonymous
        if data.onReview is not None:
            update_data["onReview"] = data.onReview

        # Always update last_updated
        update_data["last_updated"] = datetime.utcnow().isoformat()

        # Update the comment
        response = (
            supabase.table("Discussion")
            .update(update_data)
            .eq("id", comment_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update comment"
            )

        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update comment: {str(e)}"
        )


@router.delete("/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_comment(comment_id: int, current_user=Depends(get_current_user)):
    """
    Delete a comment.
    Only the comment author can delete their comment.
    """
    try:
        # First, check if comment exists and belongs to user
        existing = supabase.table("Discussion").select("*").eq("id", comment_id).execute()

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Comment not found"
            )

        if existing.data[0]["userId"] != current_user["id"]:
            raise HTTPException(
                status_code=403,
                detail="You can only delete your own comments"
            )

        # Delete the comment
        supabase.table("Discussion").delete().eq("id", comment_id).execute()

        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete comment: {str(e)}"
        )


@router.post("/{comment_id}/upvote")
def upvote_comment(comment_id: int, current_user=Depends(get_current_user)):
    """
    Upvote a comment.
    """
    try:
        # Get current vote count
        response = supabase.table("Discussion").select("upVotes").eq("id", comment_id).execute()

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Comment not found"
            )

        current_upvotes = response.data[0]["upVotes"]

        # Increment upvote
        update_response = (
            supabase.table("Discussion")
            .update({"upVotes": current_upvotes + 1})
            .eq("id", comment_id)
            .execute()
        )

        return {
            "message": "Comment upvoted successfully",
            "upVotes": update_response.data[0]["upVotes"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upvote comment: {str(e)}"
        )


@router.post("/{comment_id}/downvote")
def downvote_comment(comment_id: int, current_user=Depends(get_current_user)):
    """
    Downvote a comment.
    """
    try:
        # Get current vote count
        response = supabase.table("Discussion").select("downVotes").eq("id", comment_id).execute()

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Comment not found"
            )

        current_downvotes = response.data[0]["downVotes"]

        # Increment downvote
        update_response = (
            supabase.table("Discussion")
            .update({"downVotes": current_downvotes + 1})
            .eq("id", comment_id)
            .execute()
        )

        return {
            "message": "Comment downvoted successfully",
            "downVotes": update_response.data[0]["downVotes"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to downvote comment: {str(e)}"
        )


@router.get("/user/threads")
def get_user_discussion_threads(current_user=Depends(get_current_user)):
    """
    Get all discussion threads that the user has participated in.
    Returns complete hierarchical thread structures with user comments highlighted.
    """
    try:
        user_id = current_user["id"]
        print(f"DEBUG: Fetching threads for user_id: {user_id}")
        
        # Get all comments by the user
        user_comments = (
            supabase.table("Discussion")
            .select("id, parent, articleId, created_at")
            .eq("userId", user_id)
            .execute()
        )
        
        print(f"DEBUG: Found {len(user_comments.data) if user_comments.data else 0} user comments")
        
        if not user_comments.data:
            return {
                "threads": [],
                "count": 0
            }
        
        # Group by article and find root comments
        article_threads = {}
        
        for comment in user_comments.data:
            article_id = comment["articleId"]
            
            if article_id not in article_threads:
                article_threads[article_id] = set()
            
            # Find root comment for this comment
            root_id = comment["id"]
            parent_id = comment.get("parent")
            
            if parent_id:
                # Traverse up to find root
                current_parent = parent_id
                max_depth = 100
                depth = 0
                
                while current_parent and depth < max_depth:
                    parent_comment = (
                        supabase.table("Discussion")
                        .select("id, parent")
                        .eq("id", current_parent)
                        .execute()
                    )
                    
                    if parent_comment.data:
                        root_id = parent_comment.data[0]["id"]
                        current_parent = parent_comment.data[0].get("parent")
                        depth += 1
                    else:
                        break
            
            article_threads[article_id].add(root_id)
        
        # Build complete thread structures
        result_threads = []
        
        for article_id, root_ids in article_threads.items():
            # Get paper info
            paper_response = (
                supabase.table("Papers")
                .select("id, title, abstract")
                .eq("id", article_id)
                .execute()
            )
            
            paper_title = "Unknown Paper"
            paper_abstract = ""
            if paper_response.data:
                paper_title = paper_response.data[0].get("title", "Unknown Paper")
                paper_abstract = paper_response.data[0].get("abstract", "")
            
            # For each root comment, build the complete thread
            for root_id in root_ids:
                # Get all comments for this article
                all_article_comments = (
                    supabase.table("Discussion")
                    .select("*")
                    .eq("articleId", article_id)
                    .execute()
                )
                
                if not all_article_comments.data:
                    continue
                
                # Build comment map
                comment_map = {}
                for comment in all_article_comments.data:
                    comment_map[comment["id"]] = comment
                
                # Build hierarchical structure starting from root
                def build_thread_tree(comment_id):
                    if comment_id not in comment_map:
                        return None
                    
                    comment = comment_map[comment_id]
                    
                    # Find children
                    children = []
                    for cid, c in comment_map.items():
                        if c.get("parent") == comment_id:
                            child_tree = build_thread_tree(cid)
                            if child_tree:
                                children.append(child_tree)
                    
                    # Sort children by created_at
                    children.sort(key=lambda x: x["created_at"])
                    
                    return {
                        "id": comment["id"],
                        "userId": comment["userId"],
                        "comment": comment["comment"],
                        "created_at": comment["created_at"],
                        "upVotes": comment.get("upVotes", 0),
                        "downVotes": comment.get("downVotes", 0),
                        "isUserComment": comment["userId"] == user_id,
                        "children": children
                    }
                
                thread_tree = build_thread_tree(root_id)
                
                if thread_tree:
                    result_threads.append({
                        "article_id": article_id,
                        "paper_title": paper_title,
                        "paper_abstract": paper_abstract,
                        "root_comment_id": root_id,
                        "thread": thread_tree
                    })
        
        # Sort by most recent activity (we can use root comment creation time as proxy)
        result_threads.sort(key=lambda x: x["thread"]["created_at"], reverse=True)
        
        print(f"DEBUG: Returning {len(result_threads)} threads")
        
        return {
            "threads": result_threads,
            "count": len(result_threads)
        }
    
    except Exception as e:
        print(f"ERROR: Failed to retrieve user threads: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user threads: {str(e)}"
        )