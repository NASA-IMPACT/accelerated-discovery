import requests
import ads
import os
from models.document import Document
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
ads.config.token = os.environ["ADS_DEV_KEY"]



def search_ads(query:str,k:int=10, 
                return_metadata=True, 
                sort:str="citation_count", 
                fq:str=None,
                pydantic_output:bool=False):
    """ Sort: citation_count | year """
    ## LOOK UP HERE FOR QUERY ATTRIBUTES https://adsabs.github.io/help/search/search-syntax
    print(f"query {query} is being executed against ADS service")
    try:
        search_documents_output = list(ads.SearchQuery(q=query, sort=sort, 
                                  max_pages=1, rows=k,
                                  fq=fq,
                                  ))
    except:
        search_documents_output=[]
    output = []
    pydantic_papers = []
    for search_document in search_documents_output:
        url = None
        doi=search_document.doi[0] if search_document.doi and len(search_document.doi)>0 else None
        if "arXiv" in search_document.bibcode:
            arxiv_id_temp = search_document.bibcode.split("arXiv")[1][:-1]
            arxiv_id = arxiv_id_temp[:4] + "." + arxiv_id_temp[4:]
            #print(arxiv_id)
            url ="https://arxiv.org/pdf/" + arxiv_id
            print(url)
        else:
            if doi:
                url_res = resolve_doi(doi)
                url = url_res if not "Error" in url_res else None
                 
        if  search_document.author and len(search_document.author)>0:      
            author_string = search_document.author[0]
            if  search_document.author and len(search_document.author)>1: author_string+= " et al."
        
        pydantic_papers.append(Document(title= search_document.title[0] if search_document.title and len(search_document.title)>0 else "",
                                author=author_string,
                                url=url,
                                year=search_document.year,
                                doi=doi,
                                citation_count=search_document.citation_count
                                ))
        output_text=f"""
=============================================================================
Title: {search_document.title[0] if search_document.title else ""}
Author: {author_string}
ArXiv: {url if url else "NA"}
DOI: {search_document.doi[0] if search_document.doi and len(search_document.doi)>0 else "NA"}
Year: {search_document.year} Bibcode: {search_document.bibcode} Citations: {search_document.citation_count}
Abstract:
{search_document.abstract[:200] + "... (Truncated)" if search_document.abstract else None}
"""

        if return_metadata:
            non_hidden_attrs = {k: v for k, v in vars(search_document).items() if not k.startswith("_")}
            output.append((output_text,non_hidden_attrs))
        else: 
            output.append(output_text)
            
        #print(paper.bibcodes , paper.title, paper.abstract, paper.author, paper.year, paper.keyword)
    if pydantic_output: 
        return pydantic_papers
    else: return output
# keys = ['appl_id', 'app_filing_date', 'app_exam_name', 'public_ind', 'app_confr_number', 'inventor_name', 'app_early_pub_number', 'app_early_pub_date', 'patent_number', 'patent_issue_date', 'app_location', 'app_grp_art_number', 'last_modified', 'last_insert_time', 'patent_title', 'app_attr_dock_number', 'app_status', 'app_status_date', 'app_type', 'app_cls_sub_cls', 'corr_name', 'corr_address', 'corr_cust_no', 'transactions', 'attorneys', 'inventors', 'applicants', 'pta_pte_summary', 'pta_pte_tran_history', 'parent_continuity', 'child_continuity', 'foreign_priority', 'assignments']
# def fetch_patents(query, n:int = 3):
#     from patent_client import  USApplication
#     app = USApplication.objects.get('15710770')
    


# def search_patents(query, k:int=20):
#     google_apps = USApplication.objects.filter(first_named_applicant='Google LLC')[:3]
#     return google_apps
import http.client
import json
def search_serper_run(query,k:int=20,document_type:str = "scholar", return_readable:bool=False):   

    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": query
    })
    headers = {
    'X-API-KEY': '',
    'Content-Type': 'application/json'
    }
    conn.request("POST", f"/{document_type}", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read())
    out = []
    if document_type == "scholar":
        out_str=f"Google Scholar results for query {query}\n" + "-" * 80 + "\n"
    elif document_type == "patents":
        out_str=f"Patent search results for query {query}\n" + "-" * 80 + "\n"
    for result in data.get("organic", []):
        paper = Document(title=result.get('title',None),
                         url= result["pdfUrl"] if "pdfUrl" in result and document_type == "scholar" else result.get('link',None),
                         snippet = result.get('snippet',None),
                         citation_count=result.get("citedBy",None),
                         author= result.get("inventor",None) if  document_type == "patents" else None,
                         year=result.get("year",None) if document_type == "scholar"  else int(result.get("publicationDate",None).split("-")[0]),
                         publication_reference= result.get("publicationInfo",None),
                         )
        out_str+= paper.print_document() +"/n"
        out.append(paper)
    return out

     

def resolve_doi(doi):
    """
    Resolves a DOI to its associated resource URL.

    Parameters:
    doi (str): The Digital Object Identifier to resolve.

    Returns:
    str: The URL of the resource associated with the DOI.
    """
    #https://ui.adsabs.harvard.edu/link_gateway/2024CorRe..43.1393L/doi:10.1007/s00338-024-02545-1
    base_url = "https://doi.org/"
    url = f"{base_url}{doi}"
    try:
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            return response.url  # The final URL after redirection
        else:
            return f"Error: Received status code {response.status_code}"
    except requests.RequestException as e:
        return f"Request failed: {e}"
    
def download_file(url, save_dir="downloads"):
    """Download a file from a given URL and save it to disk with the correct extension."""
    if url:
        print(f"Downloading url {url} and saving to folder {save_dir}")
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_name = uuid4().hex
            file_path = os.path.join(save_dir, file_name)
            with open(os.path.join(save_dir, "index.csv"),"a") as f: 
                f.write(f"{file_name}\t{url}\n")

            # Save the file
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"{url} downloaded and saved as {file_path}")
            return [url, file_path]
        except:
            print(f"ERROR: {url} cannot be downloaded")
            return None
    else: return None

#print(search_serper_run("gliozzo", document_type="patents"))