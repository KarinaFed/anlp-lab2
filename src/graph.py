"""LangGraph workflow for the multi-agent system"""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import operator

from src.models import (
    RoutingDecision, TheoryExplanation, CodeHelp, StudyPlan, 
    MemoryUpdate, FinalResponse, QueryType
)
from src.agents import (
    RouterAgent, TheoryExplainerAgent, CodeHelperAgent, 
    PlannerAgent, MemoryManagerAgent
)
from src.memory import MemoryStore
from src.tools import CalculatorTool


class MultiAgentState(TypedDict):
    """State for the multi-agent system"""
    # User input
    user_query: str
    
    # Router output
    routing_decision: Optional[RoutingDecision]
    
    # Agent outputs
    theory_explanation: Optional[TheoryExplanation]
    code_help: Optional[CodeHelp]
    study_plan: Optional[StudyPlan]
    memory_update: Optional[MemoryUpdate]
    
    # Memory context
    memory_context: Optional[str]
    
    # Final output
    final_response: Optional[FinalResponse]
    
    # Control flow
    agents_involved: Annotated[List[str], operator.add]
    tools_used: Annotated[List[str], operator.add]
    error: Optional[str]


class MultiAgentSystem:
    """Multi-agent system with Router + Specialists pattern"""
    
    def __init__(self):
        self.router = RouterAgent()
        self.theory_explainer = TheoryExplainerAgent()
        self.code_helper = CodeHelperAgent()
        self.planner = PlannerAgent()
        self.memory_manager = MemoryManagerAgent()
        self.memory_store = MemoryStore()
        self.memory_manager.set_memory_store(self.memory_store)
        self.calculator = CalculatorTool()
        
        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("memory_manager", self.memory_manager_node)
        workflow.add_node("theory_explainer", self.theory_explainer_node)
        workflow.add_node("code_helper", self.code_helper_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Router routes to memory manager if needed, then to specialists
        workflow.add_conditional_edges(
            "router",
            self.route_after_router,
            {
                "memory_manager": "memory_manager",
                "theory_explainer": "theory_explainer",
                "code_helper": "code_helper",
                "planner": "planner",
                "synthesizer": "synthesizer"
            }
        )
        
        # Memory manager routes to appropriate specialist
        workflow.add_conditional_edges(
            "memory_manager",
            self.route_after_memory,
            {
                "theory_explainer": "theory_explainer",
                "code_helper": "code_helper",
                "planner": "planner",
                "synthesizer": "synthesizer"
            }
        )
        
        # All specialists go to synthesizer
        workflow.add_edge("theory_explainer", "synthesizer")
        workflow.add_edge("code_helper", "synthesizer")
        workflow.add_edge("planner", "synthesizer")
        
        # Synthesizer ends
        workflow.add_edge("synthesizer", END)
        
        return workflow
    
    async def router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Router node - classifies query and decides routing"""
        print("\n[ROUTER] Analyzing query and deciding routing...")
        
        try:
            decision = await self.router.route_query(
                state["user_query"],
                self.memory_store
            )
            
            print(f"   Query type: {decision.query_type}")
            print(f"   Target agents: {decision.target_agents}")
            print(f"   Reasoning: {decision.reasoning}")
            
            return {
                **state,
                "routing_decision": decision,
                "agents_involved": ["router"]
            }
        except Exception as e:
            print(f"   Router error: {e}")
            return {
                **state,
                "error": f"Router error: {str(e)}",
                "routing_decision": RoutingDecision(
                    query_type=QueryType.GENERAL,
                    target_agents=["theory_explainer"],
                    reasoning="Fallback routing"
                )
            }
    
    async def memory_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """Memory manager node - retrieves or stores context"""
        print("\n[MEMORY MANAGER] Managing memory...")
        
        try:
            routing = state.get("routing_decision")
            action = "retrieve" if routing and routing.needs_memory else "auto"
            
            memory_update = await self.memory_manager.manage_memory(
                state["user_query"],
                action
            )
            
            print(f"   Action: {memory_update.action}")
            if memory_update.retrieved_context:
                print(f"   Retrieved context: {memory_update.retrieved_context[:100]}...")
            
            # Update state
            new_state = {
                **state,
                "memory_update": memory_update,
                "memory_context": memory_update.retrieved_context,
                "agents_involved": ["memory_manager"]
            }
            
            # If storing, update memory store
            if memory_update.action == "store" and memory_update.value:
                self.memory_store.update_user_profile(
                    "user_preferences",
                    memory_update.value
                )
            
            return new_state
        except Exception as e:
            print(f"   Memory manager error: {e}")
            return {
                **state,
                "memory_context": None,
                "error": f"Memory error: {str(e)}"
            }
    
    async def theory_explainer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Theory explainer node"""
        print("\n[THEORY EXPLAINER] Explaining concept...")
        
        try:
            explanation = await self.theory_explainer.explain(
                state["user_query"],
                state.get("memory_context")
            )
            
            print(f"   Concept: {explanation.concept}")
            print(f"   Key points: {len(explanation.key_points)}")
            
            # Update user profile
            self.memory_store.update_user_profile("topics_asked", explanation.concept)
            
            return {
                **state,
                "theory_explanation": explanation,
                "agents_involved": ["theory_explainer"]
            }
        except Exception as e:
            print(f"   Theory explainer error: {e}")
            return {
                **state,
                "error": f"Theory explainer error: {str(e)}"
            }
    
    async def code_helper_node(self, state: MultiAgentState) -> MultiAgentState:
        """Code helper node"""
        print("\n[CODE HELPER] Helping with code...")
        
        try:
            code_help = await self.code_helper.help_with_code(
                state["user_query"],
                state.get("memory_context")
            )
            
            print(f"   Problem: {code_help.problem_description[:50]}...")
            if code_help.code_example:
                print(f"   Code example provided: {len(code_help.code_example)} chars")
            
            # Update user profile
            if "python" in state["user_query"].lower():
                self.memory_store.update_user_profile("coding_languages", "Python")
            
            tools_used = state.get("tools_used", [])
            if code_help.code_example:
                tools_used.append("code_executor")
            
            return {
                **state,
                "code_help": code_help,
                "agents_involved": ["code_helper"],
                "tools_used": tools_used
            }
        except Exception as e:
            print(f"   Code helper error: {e}")
            return {
                **state,
                "error": f"Code helper error: {str(e)}"
            }
    
    async def planner_node(self, state: MultiAgentState) -> MultiAgentState:
        """Planner node"""
        print("\n[PLANNER] Creating study plan...")
        
        try:
            user_profile = self.memory_store.get_user_profile()
            plan = await self.planner.create_plan(
                state["user_query"],
                user_profile
            )
            
            print(f"   Goal: {plan.goal}")
            print(f"   Steps: {len(plan.steps)}")
            print(f"   Total time: {plan.total_estimated_time}")
            
            # Update user profile
            self.memory_store.update_user_profile("study_goals", plan.goal)
            
            tools_used = state.get("tools_used", [])
            tools_used.append("schedule_tool")
            
            return {
                **state,
                "study_plan": plan,
                "agents_involved": ["planner"],
                "tools_used": tools_used
            }
        except Exception as e:
            print(f"   Planner error: {e}")
            return {
                **state,
                "error": f"Planner error: {str(e)}"
            }
    
    async def synthesizer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Synthesizer node - combines all agent outputs into final response"""
        print("\n[SYNTHESIZER] Synthesizing final response...")
        
        try:
            # Collect all outputs
            parts = []
            agents_used = []
            
            if state.get("theory_explanation"):
                exp = state["theory_explanation"]
                parts.append(f"## Explanation: {exp.concept}\n\n{exp.explanation}\n\n")
                parts.append(f"**Key Points:**\n" + "\n".join(f"- {p}" for p in exp.key_points) + "\n\n")
                if exp.examples:
                    parts.append(f"**Examples:**\n" + "\n".join(f"- {e}" for e in exp.examples) + "\n\n")
                agents_used.append("theory_explainer")
            
            if state.get("code_help"):
                ch = state["code_help"]
                parts.append(f"## Code Help\n\n{ch.explanation}\n\n")
                parts.append(f"**Approach:** {ch.solution_approach}\n\n")
                if ch.code_example:
                    parts.append(f"**Code Example:**\n```python\n{ch.code_example}\n```\n\n")
                if ch.best_practices:
                    parts.append(f"**Best Practices:**\n" + "\n".join(f"- {p}" for p in ch.best_practices) + "\n\n")
                agents_used.append("code_helper")
            
            if state.get("study_plan"):
                sp = state["study_plan"]
                parts.append(f"## Study Plan: {sp.goal}\n\n")
                for i, step in enumerate(sp.steps, 1):
                    parts.append(f"**Step {i}:** {step.get('description', 'N/A')}\n")
                    parts.append(f"   Estimated time: {step.get('estimated_time', 'N/A')}\n\n")
                parts.append(f"**Total estimated time:** {sp.total_estimated_time}\n\n")
                if sp.resources:
                    parts.append(f"**Resources:**\n" + "\n".join(f"- {r}" for r in sp.resources) + "\n\n")
                agents_used.append("planner")
            
            if state.get("memory_update") and state.get("memory_update").retrieved_context:
                parts.append(f"## Context from Previous Conversations\n\n{state['memory_update'].retrieved_context[:300]}...\n\n")
                agents_used.append("memory_manager")
            
            # Combine into final answer
            final_answer = "".join(parts) if parts else "I'm here to help! Could you provide more details?"
            
            # Determine confidence
            confidence = "high" if len(agents_used) > 1 and not state.get("error") else "medium"
            if state.get("error"):
                confidence = "low"
            
            final_response = FinalResponse(
                answer=final_answer,
                agents_involved=list(set(agents_used + state.get("agents_involved", []))),
                tools_used=list(set(state.get("tools_used", []))),
                memory_accessed=state.get("memory_context") is not None,
                confidence=confidence
            )
            
            # Store interaction in memory
            self.memory_store.store_interaction(
                state["user_query"],
                final_answer,
                final_response.agents_involved
            )
            
            print(f"   Final response generated")
            print(f"   Agents involved: {final_response.agents_involved}")
            print(f"   Tools used: {final_response.tools_used}")
            print(f"   Memory accessed: {final_response.memory_accessed}")
            
            return {
                **state,
                "final_response": final_response
            }
        except Exception as e:
            print(f"   Synthesizer error: {e}")
            return {
                **state,
                "error": f"Synthesizer error: {str(e)}",
                "final_response": FinalResponse(
                    answer=f"Error generating response: {str(e)}",
                    agents_involved=state.get("agents_involved", []),
                    tools_used=state.get("tools_used", []),
                    memory_accessed=False,
                    confidence="low"
                )
            }
    
    def route_after_router(self, state: MultiAgentState) -> str:
        """Route after router node"""
        routing = state.get("routing_decision")
        if not routing:
            return "synthesizer"
        
        # If memory is needed, go to memory manager first
        if routing.needs_memory:
            return "memory_manager"
        
        # Otherwise, route to first target agent
        target_agents = routing.target_agents
        if not target_agents:
            return "synthesizer"
        
        agent = target_agents[0]
        if agent == "theory_explainer":
            return "theory_explainer"
        elif agent == "code_helper":
            return "code_helper"
        elif agent == "planner":
            return "planner"
        else:
            return "synthesizer"
    
    def route_after_memory(self, state: MultiAgentState) -> str:
        """Route after memory manager node"""
        routing = state.get("routing_decision")
        if not routing:
            return "synthesizer"
        
        target_agents = routing.target_agents
        if not target_agents:
            return "synthesizer"
        
        # Route to first specialist agent
        agent = target_agents[0]
        if agent == "theory_explainer":
            return "theory_explainer"
        elif agent == "code_helper":
            return "code_helper"
        elif agent == "planner":
            return "planner"
        else:
            return "synthesizer"
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the multi-agent system"""
        initial_state = {
            "user_query": query,
            "routing_decision": None,
            "theory_explanation": None,
            "code_help": None,
            "study_plan": None,
            "memory_update": None,
            "memory_context": None,
            "final_response": None,
            "agents_involved": [],
            "tools_used": [],
            "error": None
        }
        
        result = await self.app.ainvoke(initial_state)
        return result

