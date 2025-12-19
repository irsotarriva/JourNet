import kagglehub
import numpy as np
import json
from tqdm.auto import tqdm

def main():
    # Download latest version
    print("Downloading latest version of arXiv dataset...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print("Path to dataset files:", path)
    fields_to_ignore = {"id", "versions", "authors_parsed"}
    # Count total lines (optional but gives accurate progress)
    with open(kaggleDSPath + "/arxiv-metadata-oai-snapshot.json", "r") as f:
        totalLines = sum(1 for _ in f)
    with open(kaggleDSPath + "/arxiv-metadata-oai-snapshot.json", "r") as f:
        for line in tqdm(f, total=totalLines):
            data = json.loads(line)
            paperData = {k: v for k, v in data.items() if k not in fields_to_ignore}
            authors_str = paperData.pop("authors", "")
            authors_list = [author.strip() for author in authors_str.split(",") if author.strip()]
            paperData["authors"] = authors_list

if __name__ == "__main__":
    main()