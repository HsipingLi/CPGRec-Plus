import os
import sys
import pickle
import json
import logging
import numpy as np
import torch
import aiohttp
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
import requests
import random
random.seed(3407)

import asyncio
import aiohttp
import random
import json
import torch
from tqdm import tqdm
from typing import List, Dict, Optional

import asyncio
import aiohttp
import random
import json
import torch
from tqdm import tqdm
from typing import List, Dict, Optional



class EmbeddingModel:
    def __init__(self, url: str, keys: List[str], model: str, max_concurrent_requests: int = 10):

        self.url = url
        self.keys = keys
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests

    async def _fetch_embedding(self, session: aiohttp.ClientSession, batch_text: List[str]) -> Optional[List[List[float]]]:

        payload = {
            "model": self.model,
            "input": batch_text,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {random.choice(self.keys)}",
            "Content-Type": "application/json"
        }

        try:
            async with session.post(self.url, json=payload, headers=headers) as response:
                response_data = await response.json()
                if 'data' in response_data:
                    return [item['embedding'] for item in response_data['data']]
                else:
                    print(f"Error in batch: {response_data}")
                    return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    async def _process_batch(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, batch_text: List[str]) -> Optional[List[List[float]]]:

        async with semaphore:
            return await self._fetch_embedding(session, batch_text)

    async def get_embedding(self, text: List[str], indices_none: List[int], batch_size: int = 32) -> Optional[torch.Tensor]:

        for ind in tqdm(indices_none, desc="Replacing None indices"):
            text[ind] = str({'description': 'None', 'reasoning': 'None'})

        all_embeddings = []

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in tqdm(range(0, len(text), batch_size), desc="Processing batches"):
                batch_text = text[i:i + batch_size]
                task = self._process_batch(session, semaphore, batch_text)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)

            if any(result is None for result in batch_results):
                print("Some batches failed, returning None.")
                return None

            for batch_embeddings in batch_results:
                all_embeddings.extend(batch_embeddings)

        return torch.tensor(all_embeddings)
    


