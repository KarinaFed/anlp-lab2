"""Pydantic models for structured outputs"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class QueryType(str, Enum):
    """Types of queries the system can handle"""
    THEORY = "theory"
    CODE = "code"
    PLANNING = "planning"
    GENERAL = "general"
    MEMORY = "memory"


class RoutingDecision(BaseModel):
    """Router agent output - decides which agent(s) to involve"""
    query_type: QueryType = Field(..., description="Type of query")
    target_agents: List[str] = Field(..., description="List of agent names to involve")
    reasoning: str = Field(..., description="Why these agents were chosen")
    priority: int = Field(1, description="Priority level (1=high, 2=medium, 3=low)")
    needs_memory: bool = Field(False, description="Whether memory retrieval is needed")
    needs_tools: bool = Field(False, description="Whether tool calling is needed")


class TheoryExplanation(BaseModel):
    """Theory Explainer agent output"""
    concept: str = Field(..., description="The concept being explained")
    explanation: str = Field(..., description="Detailed explanation")
    key_points: List[str] = Field(..., description="Key points to remember")
    examples: List[str] = Field(default_factory=list, description="Practical examples")
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts")
    difficulty_level: str = Field("intermediate", description="Difficulty: beginner/intermediate/advanced")


class CodeHelp(BaseModel):
    """Code Helper agent output"""
    problem_description: str = Field(..., description="Understanding of the coding problem")
    solution_approach: str = Field(..., description="Approach to solve the problem")
    code_example: Optional[str] = Field(None, description="Code example if applicable")
    explanation: str = Field(..., description="Explanation of the solution")
    best_practices: List[str] = Field(default_factory=list, description="Best practices")
    common_pitfalls: List[str] = Field(default_factory=list, description="Common pitfalls to avoid")


class StudyPlan(BaseModel):
    """Planner agent output"""
    goal: str = Field(..., description="The study goal")
    steps: List[Dict[str, Any]] = Field(..., description="List of steps with 'step', 'description', 'estimated_time'")
    total_estimated_time: str = Field(..., description="Total estimated time")
    priority_order: List[int] = Field(..., description="Order of priority for steps")
    resources: List[str] = Field(default_factory=list, description="Recommended resources")
    milestones: List[str] = Field(default_factory=list, description="Key milestones")


class MemoryUpdate(BaseModel):
    """Memory Manager agent output"""
    action: str = Field(..., description="Action: 'store', 'retrieve', 'update'")
    key: str = Field(..., description="Memory key")
    value: Optional[Any] = Field(None, description="Value to store or retrieved value")
    retrieved_context: Optional[str] = Field(None, description="Retrieved context for the query")
    reasoning: str = Field(..., description="Why this memory action was taken")


class FinalResponse(BaseModel):
    """Final response from the system"""
    answer: str = Field(..., description="Main answer to the user")
    agents_involved: List[str] = Field(..., description="List of agents that contributed")
    tools_used: List[str] = Field(default_factory=list, description="Tools that were used")
    memory_accessed: bool = Field(False, description="Whether memory was accessed")
    confidence: str = Field("medium", description="Confidence level: high/medium/low")

