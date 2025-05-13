# Read the Human annotations from the VeriScore paper.

import json
from typing import List

from akd.tools.fact_reasoner.fm_factual.context_retriever import (
    fetch_text_from_link,
    make_uniform,
)


def clean_claims(claims: List[dict], fetch_text: bool = False):
    """Clean the claims
    """
    cleaned_claims = []
    for claim_dict in claims:
        atomic_unit = claim_dict["claim"].replace("<b>", "").replace("</b>", "").replace("<br/>", "").replace("Claim:", "")
        search_results = []
        for i in range(10):
            key = f"search_result_{i+1}"
            sri = claim_dict[key].replace("<b>", "").replace("</b>", "").replace("<br/>", "\n")
            lines = sri.split("\n")
            title = None
            for line in lines[1:]:
                if line.startswith("Title:"):
                    title = line[7:]
                elif line.startswith("Content:"):
                    snippet = line[9:]
                elif line.startswith("Link:"):
                    link = line[6:]
            if title is not None:
                if fetch_text is True:
                    # Retrieve text associated with the link
                    page_text = fetch_text_from_link(link, max_size=4000)
                    doc_content = make_uniform(page_text) if len(page_text) > 0 else ""
                    search_results.append(dict(title=title, snippet=snippet, link=link, text=doc_content))
                else:
                    search_results.append(dict(title=title, snippet=snippet, link=link, text=""))
        label = claim_dict["decision_choice"]

        cleaned_claims.append(dict(claim=atomic_unit, search_results=search_results, human_label=label))

    return cleaned_claims

def read_annotations(files: List[str], fetch_text: bool = False):
    results = []
    for file in files:
        with open(file, "r") as f:
            raw_claims = json.load(f)
            results.extend(clean_claims(raw_claims, fetch_text))
    return results

if __name__ == "__main__":

    files = [
        "/home/radu/git/fm-factual/data/common_0.json",
        "/home/radu/git/fm-factual/data/common_1.json",
        "/home/radu/git/fm-factual/data/common_2.json",
        "/home/radu/git/fm-factual/data/different_1.json",
        "/home/radu/git/fm-factual/data/different_2.json",
    ]

    fetch_text = False
    data = read_annotations(files, fetch_text=fetch_text)

    print(f"Number of claims processed: {len(data)}")
    if fetch_text:
        output_filename = "/home/radu/fm-factual/data/human-labeled-google-doc.json"
    else:
        output_filename = "/home/radu/git/fm-factual/data/human-labeled-google-link.json"
    with open(output_filename, "w") as fp:
        json.dump(data, fp)
    print(f"Processed claims written to: {output_filename}")
    print("Done.")
