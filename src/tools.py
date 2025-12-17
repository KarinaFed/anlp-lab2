"""Tools available to agents"""
import re
import subprocess
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class CalculatorTool:
    """Simple calculator tool for mathematical operations"""
    
    @staticmethod
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely"""
        try:
            # Remove dangerous operations
            expression = expression.replace("import", "")
            expression = expression.replace("__", "")
            expression = expression.replace("exec", "")
            expression = expression.replace("eval", "")
            
            # Only allow basic math operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class CodeExecutorTool:
    """Tool to execute simple Python code snippets"""
    
    @staticmethod
    def execute_python(code: str, timeout: int = 5) -> Dict[str, Any]:
        """Execute Python code safely with timeout"""
        try:
            # Basic safety checks
            dangerous_keywords = ["import os", "import sys", "__import__", "open(", "file("]
            for keyword in dangerous_keywords:
                if keyword in code.lower():
                    return {
                        "success": False,
                        "output": f"Error: Potentially unsafe operation detected: {keyword}",
                        "error": "Security restriction"
                    }
            
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Execution timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }


class ScheduleTool:
    """Tool for scheduling and time management"""
    
    @staticmethod
    def parse_time_duration(duration_str: str) -> Optional[int]:
        """Parse time duration string to minutes"""
        # Examples: "2 hours", "30 minutes", "1.5 hours"
        duration_str = duration_str.lower().strip()
        
        # Extract number and unit
        match = re.search(r'(\d+\.?\d*)\s*(hour|hours|minute|minutes|min|h|m)', duration_str)
        if not match:
            return None
        
        value = float(match.group(1))
        unit = match.group(2)
        
        if unit in ['hour', 'hours', 'h']:
            return int(value * 60)
        elif unit in ['minute', 'minutes', 'min', 'm']:
            return int(value)
        
        return None
    
    @staticmethod
    def calculate_deadline(start_time: str, duration_minutes: int) -> str:
        """Calculate deadline from start time and duration"""
        try:
            start = datetime.fromisoformat(start_time)
            deadline = start + timedelta(minutes=duration_minutes)
            return deadline.isoformat()
        except:
            return "Invalid date format"
    
    @staticmethod
    def format_schedule(steps: List[Dict[str, Any]]) -> str:
        """Format a list of steps into a readable schedule"""
        schedule_lines = []
        current_time = datetime.now()
        
        for i, step in enumerate(steps, 1):
            duration = step.get('estimated_time', 'Unknown')
            description = step.get('description', 'No description')
            
            schedule_lines.append(f"{i}. {description}")
            schedule_lines.append(f"   Estimated time: {duration}")
        
        return "\n".join(schedule_lines)


class KnowledgeBaseTool:
    """Simple in-memory knowledge base for common concepts"""
    
    def __init__(self):
        self.knowledge = {
            "multi-agent system": {
                "definition": "A system composed of multiple interacting agents that work together to solve problems",
                "key_concepts": ["agents", "coordination", "communication", "distributed problem solving"],
                "examples": ["Router pattern", "Planner-executor pattern", "Supervisor pattern"]
            },
            "langgraph": {
                "definition": "A library for building stateful, multi-actor applications with LLMs",
                "key_concepts": ["state graph", "nodes", "edges", "conditional routing"],
                "examples": ["Multi-agent workflows", "Agentic applications"]
            },
            "langchain": {
                "definition": "A framework for developing applications powered by language models",
                "key_concepts": ["chains", "agents", "tools", "memory", "prompts"],
                "examples": ["RAG systems", "Agent workflows", "Tool calling"]
            }
        }
    
    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """Search knowledge base for a concept"""
        query_lower = query.lower()
        
        # Direct match
        for key, value in self.knowledge.items():
            if key in query_lower or query_lower in key:
                return value
        
        # Partial match
        for key, value in self.knowledge.items():
            if any(word in key for word in query_lower.split()):
                return value
        
        return None
    
    def add_knowledge(self, concept: str, definition: str, key_concepts: List[str]):
        """Add new knowledge to the base"""
        self.knowledge[concept.lower()] = {
            "definition": definition,
            "key_concepts": key_concepts,
            "examples": []
        }

