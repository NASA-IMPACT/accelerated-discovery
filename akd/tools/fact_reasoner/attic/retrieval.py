# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
import chromadb
import nltk
import re

from bert_score import BERTScorer
from operator import itemgetter
from typing import List
from chromadb.utils import embedding_functions

import requests
from typing import List

COLLECTION_NAME = "wikipedia_en"
DB_PATH = "/home/radu/wiki_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

# BERTScore calculation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORER = BERTScorer(model_type='bert-base-uncased', device=DEVICE)

class ChromaReader:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: str,
        collection_metadata: dict = None,
    ):
        """
        Initialize the ChromaDB.

        Args:
            collection_name: str
                The collection name in the vector database.
            persist_directory: str
                The directory used for persisting the vector database.
            embedding_model: str
                The embedding model.
            collection_metadata: dict
                A dict containing the collection metadata.
        """
        
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
            device=self.device,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )

    def is_empty(self):
        return self.collection.count() == 0

    def query(self, query_texts: str, n_results: int = 5, where_document: dict = None):
        """
        Returns the closests vector to the question vector
        
        Args:
            query_texts: str
                The user query text.
            n_results: int
                The number of results to generate.

        Returns
            The closest result to the given question.
        """
        return self.collection.query(query_texts=query_texts, n_results=n_results, where_document=where_document)
  
def postprocess(text: str) -> List[str]:
    """
    Postprocess a retrieved document by breaking it into passages.

    Args:
        text: str
            A string representing the retrieved document.

    Returns:
        A list of passages (i.e., paragraphs).
    """

    if text.count("\n\n") > 0:
        paragraphs = [p.strip() for p in text.split("\n\n")]
    else:
        paragraphs = [p.strip() for p in text.split("\n")]

    result = []
    paragraphs = [p for p in paragraphs if len(p) > 10]
    for p in paragraphs:
        subpars = [pp.strip() for pp in p.split("\n")]
        new_p = ""
        for pp in subpars:
            sentences = nltk.tokenize.sent_tokenize(pp)
            sentences = [sent for sent in sentences if len(sent)>10]
            if len(sentences) >= 0:
                new_pp = " ".join(sentences)
                new_p += " " + new_pp
        if len(new_p) > 0:
            result.append(new_p.strip())
    return result

def split_paragraphs(text: str) -> List[str]:
    """
    Postprocess a retrieved document by breaking it into paragraphs. A paragraph
    consists is a group of sentences and paragraphs are assumed to be delimited 
    by "\n\n" (2 or more new-line sequences).

    Args:
        text: str
            A string representing the retrieved document.

    Returns:
        A list of passages (i.e., paragraphs).
    """

    no_newlines = text.strip("\n")  # remove leading and trailing "\n"
    split_text = NEWLINES_RE.split(no_newlines)  # regex splitting
    chunks = [p for p in split_text if len(p.split()) > 10]

    paragraphs = []
    for chunk in chunks:
        subpars = [pp.strip() for pp in chunk.split("\n")]
        new_p = ""
        for sp in subpars:
            sentences = nltk.tokenize.sent_tokenize(sp)
            sentences = [sent for sent in sentences if len(sent.split())>5]
            if len(sentences) >= 0:
                new_pp = " ".join(sentences)
                new_p += " " + new_pp
        if len(new_p.strip()) > 0:
            paragraphs.append(new_p.strip())
    return paragraphs

def get_title(text: str) -> str:
    """
    Get the title of the retrived document. By definition, the first line in the
    document is the title (we embedded them like that).
    """
    return text[:text.find("\n")]

def scores(scorer, references: List[str], candidates: List[str]):
    P, R, F1 = scorer.score(candidates, references)
    return F1.numpy()

class ContextRetrieval:
    """
    The context retrieval service. We assume an already existing API that
    can retrieve relevant passages from a vector store (i.e., Wikipedia).
    """
    
    def __init__(
            self,
            address: str,
            port: int,
            is_remote: bool = True
    ):
        """
        Initialize the context retrieval service.

        Args:
            address: str
                The IP address of the host running the API to the vector store.
            port: int
                The port of the host running the API for querying the vector store.
            is_remote: bool
                Flag indicating a remote (http) context retrieval service.
        """
        
        self.host_address = address
        self.host_port = port
        self.is_remote = is_remote
        self.chroma = None
        # self.session = requests.Session()
        # self.session.headers.update({'Content-Type': 'application/json'})

        if not self.is_remote: # Create a local ChromaDB client
            self.chroma = ChromaReader(
                collection_name=COLLECTION_NAME, 
                persist_directory=DB_PATH, 
                embedding_model=EMBEDDING_MODEL, 
                collection_metadata={"hnsw:space": "cosine"}
            )


    def query(
            self, 
            text: str, 
            top_k: int = 1,
            n_results: int = 1,
            granularity: str = "paragraph",
            relevance: bool = False,
            where_document: dict = None
    ) -> List[str]:
        """
        Retrieve a number of contexts relevant to the input text.

        Args:
            text: str
                The input query text.

            top_k: int
                Top k most relevant passages to be retrieved.
            n_results: int
                The closest n results to be retrieved from the vector store.
            granularity: str
                The granularity of the retrieved passages (paragraph or article).
            relevance: bool
                Flag indicating that the most relevant passages are returned.
            where_document: dict
                A dict used for filtering the retrieved documents.

        Returns:
            List[dict]
                The list of retrieved contexts for the input reference. A context
                is a dict with two keys: title and text.
        """

        results = []
        if self.is_remote:
            # Send a POST request with JSON data using the session object
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            url = "http://" + self.host_address + ":" + str(self.host_port) + "/query"
            data = dict(
                query_text=text, 
                top_k=top_k, 
                n_results=n_results,
                granularity=granularity, 
                relevance=relevance,
                where_document=where_document
            )

            # Get the response
            with requests.Session() as s:
                response = s.post(url, json=data, headers=headers)

                results = []
                if response.status_code == 200: # success
                    passages = response.json()
                    # print(passages)
                    results = [passages[str(i)] for i in range(len(passages))]
                

        else: # local

            print(f"Retrieving {n_results} relevant documents for query: {text}")
            print(f"Processing retrieved documents to get top {top_k} passages")
            print(f"Granularity: {granularity}")
            print(f"Relevance: {relevance}")
            print(f"Where document: {where_document}")

            # Retrieve the relevant chunks from the vector store
            relevant_chunks = self.chroma.query(
                query_texts=[text],
                n_results=n_results,
                where_document=where_document
            )

            # Get the chunks (documents)
            docs = relevant_chunks["documents"][0]

            passages = []
            if granularity == "paragraph":
                if relevance == True:
                    print(f"Returning paragraphs with bert scores...")
                    paragraphs = []
                    for i, doc in enumerate(docs):
                        title = get_title(doc)
                        paragraphs.extend([dict(title=title, text=par) for par in split_paragraphs(text=doc)])
                    references = [text]*len(paragraphs)
                    candidates = [par["text"] for par in paragraphs]
                    try:
                        sc = scores(SCORER, references, candidates)
                    except Exception as e:
                        sc = [0.0]*len(paragraphs)
                    temp = [(sc[i], paragraphs[i]) for i in range(len(paragraphs))]
                    temp = sorted(temp, key=itemgetter(0), reverse=True)
                    passages = [p for _,p in temp]
                else:
                    print(f"Returning paragraphs without bert scores...")
                    passages = []
                    for i, doc in enumerate(docs):
                        title = get_title(doc)
                        passages.extend([dict(title=title, text=par) for par in split_paragraphs(text=doc)])
            elif granularity == "document":
                passages = [dict(title=get_title(doc), text=doc) for doc in docs]
            else:
                raise ValueError(f"Unknow granularity level: {granularity}.")

            n = len(passages) if top_k is None else min(top_k, len(passages))
            for i in range(n):
                results.append(passages[i]) # a passage is a dict with title and text as keys

        return results
    
if __name__ == '__main__':
    wikidb = ContextRetrieval(address='9.59.197.15', port=5000, is_remote=False)
    topic = "Lanny Flaherty"
    where_document = {"$contains": topic}
    contexts = wikidb.query(
        text="Who is Lanny Flaherty", 
        top_k=10,
        n_results=1,
        granularity="paragraph",
        relevance=False,
        where_document=where_document
    )
    
    print(contexts)