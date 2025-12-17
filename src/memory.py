"""Memory management for the multi-agent system"""
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.config import MEMORY_STORAGE_PATH


class MemoryStore:
    """Simple file-based memory store for session history and user context"""
    
    def __init__(self, storage_path: str = MEMORY_STORAGE_PATH):
        self.storage_path = storage_path
        self.memory: Dict[str, Any] = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading memory: {e}")
                return self._init_memory()
        return self._init_memory()
    
    def _init_memory(self) -> Dict[str, Any]:
        """Initialize empty memory structure"""
        return {
            "session_history": [],
            "user_profile": {
                "topics_asked": [],
                "coding_languages": [],
                "study_goals": []
            },
            "context": {}
        }
    
    def _save_memory(self):
        """Save memory to file"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def store_interaction(self, query: str, response: str, agents: List[str]):
        """Store an interaction in session history"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:500],  # Store first 500 chars
            "agents": agents
        }
        self.memory["session_history"].append(interaction)
        # Keep only last 20 interactions
        if len(self.memory["session_history"]) > 20:
            self.memory["session_history"] = self.memory["session_history"][-20:]
        self._save_memory()
    
    def retrieve_recent_context(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve recent interactions for context"""
        return self.memory["session_history"][-limit:]
    
    def update_user_profile(self, key: str, value: Any):
        """Update user profile information"""
        if key in self.memory["user_profile"]:
            if isinstance(self.memory["user_profile"][key], list):
                if value not in self.memory["user_profile"][key]:
                    self.memory["user_profile"][key].append(value)
            else:
                self.memory["user_profile"][key] = value
        else:
            self.memory["user_profile"][key] = value
        self._save_memory()
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        return self.memory["user_profile"]
    
    def search_history(self, keyword: str) -> List[Dict[str, Any]]:
        """Search session history for keyword"""
        results = []
        keyword_lower = keyword.lower()
        for interaction in self.memory["session_history"]:
            if (keyword_lower in interaction["query"].lower() or 
                keyword_lower in interaction["response"].lower()):
                results.append(interaction)
        return results
    
    def clear_memory(self):
        """Clear all memory (for testing)"""
        self.memory = self._init_memory()
        self._save_memory()

