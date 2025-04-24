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

# Search API

import os
import torch
import json
import requests

from ast import literal_eval
from dotenv import load_dotenv

class SearchAPI():

    def __init__(self, cache_dir: str):
        # invariant variables
        load_dotenv(override=True)
        self.serper_key = os.getenv("SERPER_API_KEY")
        self.url = "https://google.serper.dev/search"
        self.headers = {'X-API-KEY': self.serper_key,
                        'Content-Type': 'application/json'}
        # cache related
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "search_cache.json")
            self.cache_dict = self.load_cache()
        else: # no caching
            self.cache_file = None
            self.cache_dict = {}
        self.add_n = 0
        self.save_interval = 10

    def get_snippets(self, claim_lst):
        text_claim_snippets_dict = {}
        for query in claim_lst:
            search_result = self.get_search_res(query)
            if "statusCode" in search_result:  # and search_result['statusCode'] == 403:
                print(search_result['message'])
                exit()
            if "organic" in search_result:
                organic_res = search_result["organic"]
            else:
                organic_res = []

            search_res_lst = []
            for item in organic_res:
                title = item["title"] if "title" in item else ""
                snippet = item["snippet"] if "snippet" in item else ""
                link = item["link"] if "link" in item else ""
                    
                search_res_lst.append({"title": title,
                                       "snippet": snippet,
                                       "link": link})
            text_claim_snippets_dict[query] = search_res_lst
        return text_claim_snippets_dict

    def get_search_res(self, query):
        # check if prompt is in cache; if so, return from cache
        cache_key = query.strip()
        if cache_key in self.cache_dict:
            # print("Getting search results from cache ...")
            return self.cache_dict[cache_key]

        payload = json.dumps({"q": query})
        response = requests.request("POST",
                                    self.url,
                                    headers=self.headers,
                                    data=payload)
        response_json = literal_eval(response.text)

        # update cache
        self.cache_dict[query.strip()] = response_json
        self.add_n += 1

        # save cache every save_interval times
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        return response_json

    def save_cache(self):
        # load the latest cache first, since if there were other processes 
        # running in parallel, cache might have been updated
        if self.cache_file is None: return
        cache = self.load_cache().items()
        for k, v in cache:
            self.cache_dict[k] = v
        print(f"[SearchAPI] Saving search cache ...")
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # load a json file
                cache = json.load(f)
                print(f"[SearchAPI] Loading cache ...")
        else:
            cache = {}
        return cache

if __name__ == '__main__':

    cache_dir = "/home/radu/data/cache"

    # text = "Neil B. Todd was an American geneticist"
    text = "Lanny Flaherty is an American."

    # Initialize search api
    web_search = SearchAPI(cache_dir=None)

    claim_lst = [text]
    claim_snippets = web_search.get_snippets(claim_lst)
    print(claim_snippets)
    print("Done.")
