"""Agent implementations for the multi-agent system"""
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
import asyncio
import json

from src.models import (
    RoutingDecision, TheoryExplanation, CodeHelp, StudyPlan, 
    MemoryUpdate, QueryType
)
from src.config import BASE_URL, API_KEY, MODEL_NAME
from src.memory import MemoryStore
from src.tools import CalculatorTool, CodeExecutorTool, ScheduleTool, KnowledgeBaseTool


# Initialize LLMs with different configurations
def create_llm(temperature: float = 0.7, role: str = "assistant"):
    """Create LLM instance with retry logic"""
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=temperature,
        max_retries=3,  # Retry logic for Pydantic parsing
        timeout=60
    )


class RouterAgent:
    """Router agent - classifies queries and decides which agents to involve"""
    
    def __init__(self):
        self.llm = create_llm(temperature=0.1, role="router")
        self.parser = PydanticOutputParser(pydantic_object=RoutingDecision)
    
    async def route_query(self, query: str, memory_store: MemoryStore) -> RoutingDecision:
        """Analyze query and decide routing"""
        # Get recent context
        recent_context = memory_store.retrieve_recent_context(limit=2)
        context_str = ""
        if recent_context:
            context_str = "\nRecent interactions:\n"
            for ctx in recent_context:
                context_str += f"- Q: {ctx['query'][:100]}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a router agent that analyzes user queries and decides which specialized agents should handle them.

Available agents:
- theory_explainer: For questions about concepts, theories, explanations (e.g., "What is X?", "Explain Y", "How does Z work?")
- code_helper: For coding questions, debugging, implementation help (e.g., "How to implement X?", "Fix this code", "Python syntax")
- planner: For planning, scheduling, study plans, task breakdown (e.g., "Create a study plan", "How to learn X?", "Schedule for Y")
- memory_manager: For questions about previous conversations, context retrieval (e.g., "What did we discuss?", "Remember X")

Classify the query and decide:
1. query_type: theory, code, planning, general, or memory
2. target_agents: List of agent names to involve (can be multiple)
3. reasoning: Why these agents were chosen
4. needs_memory: Whether to retrieve previous context
5. needs_tools: Whether tools might be needed (calculator, code executor, etc.)

{format_instructions}"""),
            ("human", """User query: {query}
{context}

Analyze and route this query.""")
        ])
        
        try:
            chain = prompt | self.llm | self.parser
            decision = await chain.ainvoke({
                "query": query,
                "context": context_str,
                "format_instructions": self.parser.get_format_instructions()
            })
            return decision
        except Exception as e:
            print(f"Router error: {e}")
            # Fallback routing
            query_lower = query.lower()
            if any(word in query_lower for word in ["what is", "explain", "concept", "theory", "how does"]):
                query_type = QueryType.THEORY
                target_agents = ["theory_explainer"]
            elif any(word in query_lower for word in ["code", "implement", "python", "function", "debug", "syntax"]):
                query_type = QueryType.CODE
                target_agents = ["code_helper"]
            elif any(word in query_lower for word in ["plan", "schedule", "learn", "study", "steps"]):
                query_type = QueryType.PLANNING
                target_agents = ["planner"]
            else:
                query_type = QueryType.GENERAL
                target_agents = ["theory_explainer"]
            
            return RoutingDecision(
                query_type=query_type,
                target_agents=target_agents,
                reasoning="Fallback routing based on keywords",
                needs_memory=len(recent_context) > 0,
                needs_tools=False
            )


class TheoryExplainerAgent:
    """Agent that explains theoretical concepts"""
    
    def __init__(self):
        self.llm = create_llm(temperature=0.7, role="theory_explainer")
        self.parser = PydanticOutputParser(pydantic_object=TheoryExplanation)
        self.knowledge_base = KnowledgeBaseTool()
    
    async def explain(self, query: str, context: Optional[str] = None) -> TheoryExplanation:
        """Explain a theoretical concept"""
        # Check knowledge base first
        kb_result = self.knowledge_base.search(query)
        kb_context = ""
        if kb_result:
            kb_context = f"\nKnowledge base entry:\n{json.dumps(kb_result, indent=2)}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educator specializing in explaining complex concepts clearly and comprehensively.

Your task is to explain concepts in a structured way:
- concept: The main concept being explained
- explanation: Detailed, clear explanation (2-3 paragraphs)
- key_points: 3-5 key points to remember
- examples: Practical examples to illustrate the concept
- related_concepts: Related concepts the user might want to explore
- difficulty_level: beginner, intermediate, or advanced

{format_instructions}"""),
            ("human", """User query: {query}
{context}
{kb_context}

Provide a comprehensive explanation.""")
        ])
        
        try:
            chain = prompt | self.llm | self.parser
            explanation = await chain.ainvoke({
                "query": query,
                "context": context or "",
                "kb_context": kb_context,
                "format_instructions": self.parser.get_format_instructions()
            })
            return explanation
        except Exception as e:
            print(f"Theory explainer error: {e}")
            # Fallback
            return TheoryExplanation(
                concept=query.split("?")[0] if "?" in query else query[:50],
                explanation="I'll help explain this concept. Let me provide a clear explanation...",
                key_points=["Key point 1", "Key point 2"],
                examples=[],
                related_concepts=[],
                difficulty_level="intermediate"
            )


class CodeHelperAgent:
    """Agent that helps with coding questions"""
    
    def __init__(self):
        self.llm = create_llm(temperature=0.3, role="code_helper")
        self.parser = PydanticOutputParser(pydantic_object=CodeHelp)
        self.code_executor = CodeExecutorTool()
    
    async def help_with_code(self, query: str, context: Optional[str] = None) -> CodeHelp:
        """Help with coding questions"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert programming assistant specializing in code explanation, debugging, and best practices.

Your task is to help with coding questions:
- problem_description: Your understanding of the coding problem
- solution_approach: Step-by-step approach to solve it
- code_example: Working code example (if applicable)
- explanation: Detailed explanation of the solution
- best_practices: 3-5 best practices related to this problem
- common_pitfalls: Common mistakes to avoid

{format_instructions}"""),
            ("human", """User query: {query}
{context}

Provide comprehensive coding help.""")
        ])
        
        try:
            chain = prompt | self.llm | self.parser
            code_help = await chain.ainvoke({
                "query": query,
                "context": context or "",
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # If there's code to execute, try it
            if code_help.code_example and "print" in code_help.code_example.lower():
                # Extract and execute simple code snippets
                exec_result = self.code_executor.execute_python(code_help.code_example)
                if exec_result["success"]:
                    code_help.explanation += f"\n\nExecution result: {exec_result['output']}"
            
            return code_help
        except Exception as e:
            print(f"Code helper error: {e}")
            return CodeHelp(
                problem_description=query,
                solution_approach="Analyze the problem step by step",
                code_example=None,
                explanation="I'll help you with this coding question.",
                best_practices=[],
                common_pitfalls=[]
            )


class PlannerAgent:
    """Agent that creates study plans and schedules"""
    
    def __init__(self):
        self.llm = create_llm(temperature=0.5, role="planner")
        self.parser = PydanticOutputParser(pydantic_object=StudyPlan)
        self.schedule_tool = ScheduleTool()
    
    async def create_plan(self, query: str, user_profile: Optional[Dict[str, Any]] = None) -> StudyPlan:
        """Create a study plan"""
        profile_context = ""
        if user_profile:
            profile_context = f"\nUser profile:\n- Topics asked before: {user_profile.get('topics_asked', [])}\n"
            profile_context += f"- Coding languages: {user_profile.get('coding_languages', [])}\n"
            profile_context += f"- Study goals: {user_profile.get('study_goals', [])}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert study planner and productivity coach.

Your task is to create structured study plans:
- goal: The study goal extracted from the query
- steps: List of steps, each with 'step' (number), 'description', 'estimated_time' (e.g., "2 hours", "30 minutes")
- total_estimated_time: Total time estimate
- priority_order: Order of steps by priority (list of step numbers)
- resources: Recommended learning resources
- milestones: Key milestones to track progress

{format_instructions}"""),
            ("human", """User query: {query}
{profile_context}

Create a comprehensive study plan.""")
        ])
        
        try:
            chain = prompt | self.llm | self.parser
            plan = await chain.ainvoke({
                "query": query,
                "profile_context": profile_context,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Format schedule using tool
            schedule_text = self.schedule_tool.format_schedule(plan.steps)
            
            return plan
        except Exception as e:
            print(f"Planner error: {e}")
            return StudyPlan(
                goal=query,
                steps=[
                    {"step": 1, "description": "Understand basics", "estimated_time": "2 hours"},
                    {"step": 2, "description": "Practice examples", "estimated_time": "3 hours"}
                ],
                total_estimated_time="5 hours",
                priority_order=[1, 2],
                resources=[],
                milestones=[]
            )


class MemoryManagerAgent:
    """Agent that manages memory and context"""
    
    def __init__(self):
        self.llm = create_llm(temperature=0.2, role="memory_manager")
        self.parser = PydanticOutputParser(pydantic_object=MemoryUpdate)
        self.memory_store: Optional[MemoryStore] = None
    
    def set_memory_store(self, memory_store: MemoryStore):
        """Set the memory store instance"""
        self.memory_store = memory_store
    
    async def manage_memory(self, query: str, action: str = "auto") -> MemoryUpdate:
        """Manage memory based on query"""
        if not self.memory_store:
            return MemoryUpdate(
                action="none",
                key="",
                value=None,
                retrieved_context=None,
                reasoning="Memory store not available"
            )
        
        # Determine action
        query_lower = query.lower()
        if action == "auto":
            if any(word in query_lower for word in ["remember", "store", "save"]):
                action = "store"
            elif any(word in query_lower for word in ["what did", "previous", "earlier", "before"]):
                action = "retrieve"
            else:
                action = "retrieve"  # Default to retrieve for context
        
        if action == "retrieve":
            # Retrieve relevant context
            recent = self.memory_store.retrieve_recent_context(limit=3)
            context_parts = []
            for ctx in recent:
                context_parts.append(f"Q: {ctx['query']}\nA: {ctx['response'][:200]}")
            
            retrieved_context = "\n".join(context_parts) if context_parts else "No previous context found"
            
            # Also search for keywords
            keywords = [word for word in query_lower.split() if len(word) > 3]
            search_results = []
            for keyword in keywords[:2]:
                results = self.memory_store.search_history(keyword)
                search_results.extend(results)
            
            if search_results:
                retrieved_context += "\n\nRelated previous discussions:\n"
                for result in search_results[:2]:
                    retrieved_context += f"- {result['query'][:100]}\n"
            
            return MemoryUpdate(
                action="retrieve",
                key="session_context",
                value=None,
                retrieved_context=retrieved_context,
                reasoning="Retrieved recent session history and related discussions"
            )
        
        elif action == "store":
            # Extract key information to store
            return MemoryUpdate(
                action="store",
                key="user_preference",
                value=query,
                retrieved_context=None,
                reasoning="Storing user preference or information"
            )
        
        return MemoryUpdate(
            action="none",
            key="",
            value=None,
            retrieved_context=None,
            reasoning="No memory action needed"
        )

