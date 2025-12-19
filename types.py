import datetime
class Author:
    def __init__(self, name: str, affiliations: list[int] = None, user_id: int = None, id: int = None):
        self.id = id
        self.name = name
        self.user_id = user_id
        self.affiliation = affiliations

class Institution:
    def __init__(self, name: str, id: int = None):
        self.id = id
        self.name = name

class Paper:
    def __init__(self, title: str, submitter: Author = Author("Anonymous"), authors: list[Author] = [], journal_ref: str = "", doi: str = "", report_number: str = "", categories: list[str] = "", paper_license: str = "", abstract: str = "", updated_date: datetime = None, comments: str = "", id:int = None, discussions: list[int] = []):
        self.id = id
        self.title = title
        self.submitter = submitter
        self.authors = authors
        self.journal_ref = journal_ref
        self.doi = doi
        self.report_number = report_number
        self.categories = categories
        self.license = paper_license
        self.abstract = abstract
        self.updated_date = updated_date
        self.comments = comments
        self.discussions = discussions

class Comment:
    def __init__(self, userId: int, paper_id: int, id: int = None, comment : str = "", isAnonymous: bool = False, onReview: bool = False, upvotes: int = 0, downvotes: int = 0, articleId: int = None, created_at: datetime = None, summary: str = "", last_updated: datetime = None, parentCommentId: int = None):
        self.id = id
        self.parentCommentId = parentCommentId
        self.userId = userId
        self.paper_id = paper_id
        self.comment = comment
        self.isAnonymous = isAnonymous
        self.onReview = onReview
        self.upvotes = upvotes
        self.downvotes = downvotes
        self.articleId = articleId
        self.created_at = created_at
        self.summary = summary
        self.last_updated = last_updated

class 
