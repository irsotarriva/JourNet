import datetime
class Author:
    def __init__(self, name: str, affiliation: str = None, user_id: int = None):
        self.user_id = user_id
        self.name = name
        self.affiliation = affiliation
class Paper:
    def __init__(self, uuid: int, title: str, submitter: Author = Author("Anonymous"), authors: list[Author] = [], journal_ref: str = "", doi: str = "", report_number: str = "", categories: list[str] = "", paper_license: str = "", abstract: str = "", updated_date: datetime = None, comments: str = ""):
        self.uuid = uuid
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