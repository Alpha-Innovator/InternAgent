from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
from openai import OpenAI
import os
import requests

def call_model(query, model_name, key, url):
    if len(query) > 300000:
        query = query[:300000]

    client = OpenAI(
        base_url=url,
        api_key=key,
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=model_name,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                },
            ]
            }
        ]
    )
    return completion.choices[0].message.content


# class AKBClient:
#     def __init__(self, base_url="http://localhost:9000"):
#         self.base_url = base_url
#         self.session = requests.Session()
#         self.session.headers.update({"Content-Type": "application/json"})
    
#     def hybrid_search(self, query: str, top_k: int = 5, weights: Dict[str, float] = None) -> List[Dict]:
#         endpoint = f"{self.base_url}/search/hybrid"
#         payload = {
#             "query": query,
#             "top_k": top_k,
#             "weights": weights or {"text": 0.5, "semantic": 0.5}
#         }
        
#         try:
#             response = self.session.post(endpoint, json=payload)
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.RequestException as e:
#             print(f"Hybrid search error: {str(e)}")
#             return []

#     def text_search(self, query: str, top_k: int = 5) -> List[Dict]:
#         endpoint = f"{self.base_url}/search/text"
#         payload = {"query": query, "top_k": top_k}
        
#         try:
#             response = self.session.post(endpoint, json=payload)
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.RequestException as e:
#             print(f"Text search error: {str(e)}")
#             return []

#     def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
#         endpoint = f"{self.base_url}/search/semantic"
#         payload = {"query": query, "top_k": top_k}
        
#         try:
#             response = self.session.post(endpoint, json=payload)
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.RequestException as e:
#             print(f"Semantic search error: {str(e)}")
#             return []
class AKBClient:
    def __init__(self, base_url="http://localhost:9000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def _post_no_proxy(self, endpoint: str, payload: dict) -> List[Dict]:
        """在请求时临时取消代理"""
        # 保存原代理
        old_http_proxy = os.environ.get("http_proxy")
        old_https_proxy = os.environ.get("https_proxy")

        try:
            # 临时取消代理
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)

            response = self.session.post(endpoint, json=payload, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request to {endpoint} failed: {str(e)}")
            return []
        finally:
            # 恢复原代理
            if old_http_proxy is not None:
                os.environ["http_proxy"] = old_http_proxy
            if old_https_proxy is not None:
                os.environ["https_proxy"] = old_https_proxy

    def hybrid_search(self, query: str, top_k: int = 5, weights: Dict[str, float] = None) -> List[Dict]:
        endpoint = f"{self.base_url}/search/hybrid"
        payload = {
            "query": query,
            "top_k": top_k,
            "weights": weights or {"text": 0.5, "semantic": 0.5}
        }
        return self._post_no_proxy(endpoint, payload)

    def text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/text"
        payload = {"query": query, "top_k": top_k}
        return self._post_no_proxy(endpoint, payload)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        endpoint = f"{self.base_url}/search/semantic"
        payload = {"query": query, "top_k": top_k}
        return self._post_no_proxy(endpoint, payload)