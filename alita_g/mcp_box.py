import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings


@dataclass
class MCPItem:
    name: str
    code: str
    description: str
    use_case: str
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


class MCPBox:
    def __init__(self, storage_path: str = "mcp_box.json"):
        self.storage_path = storage_path
        self.items: List[MCPItem] = []
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.load()

    def add_item(self, name: str, code: str, description: str, use_case: str) -> None:
        # Concatenate description and use case for embedding as per paper
        context = f"{description}\n{use_case}"
        embedding = self.embeddings_model.embed_query(context)
        item = MCPItem(name, code, description, use_case, embedding)
        self.items.append(item)
        self.save()

    def save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump([item.to_dict() for item in self.items], f, indent=2)

    def load(self) -> None:
        """Loads MCP items from storage with error handling."""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Incorrect format in MCP box storage")
                self.items = [MCPItem(**item) for item in data]
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not load MCPBox from {self.storage_path}: {e}")
            self.items = []

    def retrieve(
        self, query: str, threshold: float = 0.7, top_k: Optional[int] = None
    ) -> List[MCPItem]:
        if not self.items:
            return []

        query_embedding = np.array(self.embeddings_model.embed_query(query))
        item_embeddings = np.array([item.embedding for item in self.items if item.embedding])

        if len(item_embeddings) == 0:
            return []

        # Cosine similarity
        similarities = np.dot(item_embeddings, query_embedding) / (
            np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        results = []
        for i, sim in enumerate(similarities):
            if top_k is None:
                if sim >= threshold:
                    results.append((self.items[i], sim))
            else:
                results.append((self.items[i], sim))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            results = results[:top_k]
            # Even in top-k, paper suggests a minimum threshold can be useful,
            # but Algorithm 1 shows them as separate strategies.

        return [item for item, sim in results]
