"""
Long-term memory module for storing learned skills.
Manages CRUD operations on skills.json database.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class SkillDatabase:
    """Manages long-term skill storage and retrieval."""

    def __init__(self, skills_file: str = "skills.json"):
        """
        Initialize skill database.

        Args:
            skills_file: Path to JSON file storing skills
        """
        self.skills_file = os.path.join(os.path.dirname(__file__), "data", skills_file)
        self.skills = self._load_skills()

    def _load_skills(self) -> Dict:
        """Load skills from JSON file or create empty database."""
        if os.path.exists(self.skills_file):
            with open(self.skills_file, "r") as f:
                return json.load(f)
        return {"skills": []}

    def _save_skills(self):
        """Persist skills to JSON file."""
        os.makedirs(os.path.dirname(self.skills_file), exist_ok=True)
        with open(self.skills_file, "w") as f:
            json.dump(self.skills, f, indent=2)

    def add_skill(
        self,
        skill_name: str,
        skill_description: str,
        skill_code: str,
        task_context: str,
        success_count: int = 1,
    ) -> str:
        """
        Add new skill to database.

        Args:
            skill_name: Unique identifier for skill
            skill_description: Natural language description
            skill_code: Python code as string (uses smolagents tools)
            task_context: Original task where skill was learned
            success_count: Initial success count

        Returns:
            skill_id of added skill
        """
        # Generate unique skill_id
        skill_id = f"skill_{len(self.skills['skills'])}"

        # Check if skill name already exists
        existing = self.get_skill_by_name(skill_name)
        if existing:
            # Update existing skill instead
            return self.update_skill(
                existing["skill_id"], skill_code=skill_code, increment_success=True
            )

        skill = {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "description": skill_description,
            "code": skill_code,
            "task_context": task_context,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "success_count": success_count,
            "usage_count": 1,
        }

        self.skills["skills"].append(skill)
        self._save_skills()
        return skill_id

    def get_skill(self, skill_id: str) -> Optional[Dict]:
        """
        Retrieve skill by ID.

        Args:
            skill_id: Unique skill identifier

        Returns:
            Skill dict or None if not found
        """
        for skill in self.skills["skills"]:
            if skill["skill_id"] == skill_id:
                return skill.copy()
        return None

    def get_skill_by_name(self, skill_name: str) -> Optional[Dict]:
        """
        Retrieve skill by name.

        Args:
            skill_name: Skill name

        Returns:
            Skill dict or None if not found
        """
        for skill in self.skills["skills"]:
            if skill["skill_name"] == skill_name:
                return skill.copy()
        return None

    def get_all_skills(self) -> List[Dict]:
        """
        Get all skills in database.

        Returns:
            List of all skill dicts
        """
        return [skill.copy() for skill in self.skills["skills"]]

    def update_skill(
        self,
        skill_id: str,
        skill_code: Optional[str] = None,
        increment_success: bool = False,
    ) -> str:
        """
        Update existing skill.

        Args:
            skill_id: Skill to update
            skill_code: New code (optional)
            increment_success: Whether to increment success_count

        Returns:
            skill_id if successful, empty string if not found
        """
        for skill in self.skills["skills"]:
            if skill["skill_id"] == skill_id:
                if skill_code is not None:
                    skill["code"] = skill_code
                if increment_success:
                    skill["success_count"] += 1
                skill["usage_count"] += 1
                skill["last_used"] = datetime.now().isoformat()
                self._save_skills()
                return skill_id
        return ""

    def delete_skill(self, skill_id: str) -> bool:
        """
        Delete skill from database.

        Args:
            skill_id: Skill to delete

        Returns:
            True if deleted, False if not found
        """
        original_length = len(self.skills["skills"])
        self.skills["skills"] = [
            s for s in self.skills["skills"] if s["skill_id"] != skill_id
        ]

        if len(self.skills["skills"]) < original_length:
            self._save_skills()
            return True
        return False

    def search_skills_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Simple keyword search in skill names and descriptions.

        Args:
            keyword: Search term

        Returns:
            List of matching skills
        """
        keyword_lower = keyword.lower()
        results = []

        for skill in self.skills["skills"]:
            if (
                keyword_lower in skill["skill_name"].lower()
                or keyword_lower in skill["description"].lower()
            ):
                results.append(skill.copy())

        return results

    def get_top_skills(self, n: int = 10) -> List[Dict]:
        """
        Get top N most successful skills.

        Args:
            n: Number of skills to return

        Returns:
            List of top skills sorted by success_count
        """
        sorted_skills = sorted(
            self.skills["skills"], key=lambda s: s["success_count"], reverse=True
        )
        return [skill.copy() for skill in sorted_skills[:n]]

    def get_skill_count(self) -> int:
        """Get total number of skills in database."""
        return len(self.skills["skills"])
