"""
Skill retrieval module using vector embeddings for semantic search.
Finds relevant skills for current task to reduce LLM token usage.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    """Semantic skill retrieval using vector embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_file: str = "skill_embeddings.npy",
        skills_file: str = "skills.json",
    ):
        """
        Initialize embedding retriever.

        Args:
            model_name: SentenceTransformer model name
            embeddings_file: Path to cached embeddings
            skills_file: Path to skills database
        """
        self.model = SentenceTransformer(model_name)
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.embeddings_file = os.path.join(self.data_dir, embeddings_file)
        self.skills_file = os.path.join(self.data_dir, skills_file)

        self.skill_embeddings = None
        self.skill_ids = []
        self._load_or_create_embeddings()

    def _load_or_create_embeddings(self):
        """Load cached embeddings or create new ones."""
        if os.path.exists(self.embeddings_file):
            data = np.load(self.embeddings_file, allow_pickle=True).item()
            self.skill_embeddings = data.get("embeddings")
            self.skill_ids = data.get("skill_ids", [])
        else:
            self.rebuild_embeddings()

    def rebuild_embeddings(self):
        """
        Rebuild embeddings from skills.json.
        Call this when skills are added/modified.
        """
        if not os.path.exists(self.skills_file):
            # No skills yet, create empty cache
            self.skill_embeddings = np.array([])
            self.skill_ids = []
            return

        with open(self.skills_file, "r") as f:
            skills_data = json.load(f)

        skills = skills_data.get("skills", [])
        if not skills:
            self.skill_embeddings = np.array([])
            self.skill_ids = []
            return

        # Create embeddings from skill descriptions
        texts = [f"{skill['skill_name']}: {skill['description']}" for skill in skills]
        self.skill_ids = [skill["skill_id"] for skill in skills]

        self.skill_embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        # Save cache
        os.makedirs(self.data_dir, exist_ok=True)
        np.save(
            self.embeddings_file,
            {"embeddings": self.skill_embeddings, "skill_ids": self.skill_ids},
        )

    def retrieve_relevant_skills(self, query: str, top_k: int = 5) -> List[str]:
        """
        Find most relevant skills for given query.

        Args:
            query: Task description or current subtask
            top_k: Number of skills to retrieve

        Returns:
            List of skill_ids ranked by relevance
        """
        if self.skill_embeddings is None or len(self.skill_embeddings) == 0:
            return []

        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.skill_embeddings)

        # Get top_k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.skill_ids[idx] for idx in top_indices]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query vector and skill vectors.

        Args:
            vec1: Query embedding (1D)
            vec2: Skill embeddings (2D)

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-10)

        # Dot product
        return np.dot(vec2_norm, vec1_norm)

    def get_skill_count(self) -> int:
        """Get number of skills in embedding cache."""
        return len(self.skill_ids)

    def invalidate_cache(self):
        """Force rebuild on next retrieval."""
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)
        self.skill_embeddings = None
        self.skill_ids = []
