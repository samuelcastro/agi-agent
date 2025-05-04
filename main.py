#!/usr/bin/env python3
import dataclasses
from typing import Dict, Tuple, Union, Optional, List

from agisdk import REAL

import base64
import dataclasses
import numpy as np
import io
import logging
import os
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from PIL import Image
from typing import Literal

# Import str2bool function for boolean command line arguments
from agisdk.REAL.demo_agent.run_demo import str2bool

from agisdk.REAL.browsergym.experiments import Agent, AbstractAgentArgs
from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
from agisdk.REAL.browsergym.core.action.python import PythonActionSet
from agisdk.REAL.browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

# Import the AgentLogger class
import sys
import os
# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from rl_training.agents.agent_logger_class import AgentLogger

# Configure logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic.v1 import BaseModel, Field # <-- Use pydantic.v1 compatibility
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated, Optional
from langchain_openai import ChatOpenAI # Ensure OpenAI is imported correctly

# Define Pydantic models for structured LLM output
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing in the proposed action or thought process.")
    superfluous: str = Field(description="Critique of what is superfluous or incorrect in the proposed action or thought process.")

class ActionDecision(BaseModel):
    """The final decision on the action to take, including self-reflection."""
    action: str = Field(description="The action string to execute in the environment (e.g., 'click(\"12\")').")
    reflection: Reflection = Field(description="Your reflection on the reasoning and chosen action.")

# --- New Models for Multi-Step Reflexion --- 
class ProposedAction(BaseModel):
    """A proposed action with reasoning."""
    action: str = Field(description="The proposed action string (e.g., 'click(\"12\")').")
    reasoning: str = Field(description="Step-by-step reasoning for proposing this action.")

class Critique(BaseModel):
    """A critique of the proposed action."""
    critique: str = Field(description="Constructive critique of the proposed action's reasoning and applicability.")
    is_sufficient: bool = Field(description="Whether the proposed action is sufficient and correct to proceed (True/False).")
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")
# --- End New Models --- 

# --- Supervisor Model --- 
class RoutingDecision(BaseModel):
    """Decision on the next step in the agent's internal workflow."""
    next_step: Literal["propose", "critique", "revise", "force_infeasible", "final_answer"] = Field(
        description="The next node or action to take: 'propose' action, 'critique' proposal, 'revise' proposal, 'force_infeasible' if stuck/failed, or 'final_answer' if the last proposal is good."
    )
    reasoning: str = Field(description="Brief reasoning for the routing decision.")
# --- End Supervisor Model --- 


# Define LangGraph state
class AgentState(TypedDict):
    # Base context (System, Goal, Obs, History) constructed once per get_action call
    initial_messages: List[BaseMessage] 
    # Running message list for the current internal graph execution
    messages: Annotated[List[BaseMessage], add_messages]
    # Fields updated by the graph nodes
    proposed_action: Optional[ProposedAction] = None
    critique: Optional[Critique] = None
    supervisor_decision: Optional[RoutingDecision] = None # Added field for supervisor output
    revision_attempts: int = 0 


# Handling Screenshots
def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class DemoAgent(Agent):
    """A basic agent using OpenAI API, adapted with LangGraph Reflexion."""

    def obs_preprocessor(self, obs: dict) -> dict:
        # Keep this as is for now, it prepares data for the prompt
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
            # Pass raw obs if needed by LangGraph nodes later
            "raw_obs": obs 
        }
        
    def reset(self):
        """Called when the environment is reset"""
        super().reset()
        # Reset action and reflection history
        self.action_history = []
        self.reflection_history: List[Reflection] = [] # Store reflection objects

    def close(self):
        """Called when the agent is being closed"""
        # Complete the agent logger session if available
        if hasattr(self, 'agent_logger') and self.agent_logger is not None:
            try:
                session_id = self.agent_logger.complete()
                print(f"Agent logger session completed with ID: {session_id}")
            except Exception as e:
                logger.error(f"Failed to complete agent logger session: {e}")
                
        super().close()

    def __init__(
        self,
        model_name: str, 
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
        # system_message_handling: Literal["separate", "combined"] = "separate", # Let LangGraph manage messages
    ) -> None:
        # super().__init__() # Call Agent.__init__ later after setting up graph
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        # self.system_message_handling = system_message_handling # Removed

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        # Use langchain_openai ChatOpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment, using a dummy key")
            openai_api_key = "sk-dummy-key-for-testing"
        
        # Use the model specified, defaulting to gpt-4o if needed
        self.model_name = model_name or "gpt-4o"
        self.llm = ChatOpenAI(model=self.model_name, api_key=openai_api_key, temperature=0) # Use ChatOpenAI

        # Bind the ActionDecision tool to the LLM
        self.llm_with_tool = self.llm.bind_tools(tools=[ActionDecision], tool_choice=ActionDecision.__name__)
        self.parser = PydanticToolsParser(tools=[ActionDecision])

        # --- LangGraph Setup ---
        builder = StateGraph(AgentState)

        # Define Nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("propose_action", self._propose_action_node)
        builder.add_node("critique_action", self._critique_action_node)
        builder.add_node("revise_action", self._revise_action_node)
        # Potentially add a node for forced infeasibility if needed, or handle in supervisor

        # Define Edges
        builder.add_edge(START, "supervisor") # Start with the supervisor
        
        # Supervisor routes to propose, revise, or ends graph
        builder.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "propose": "propose_action",
                "revise": "revise_action", # Allow supervisor to directly trigger revision
                "force_infeasible": END, # Supervisor decision handled in get_action
                "final_answer": END # Supervisor decision handled in get_action
            }
        )
        
        # Propose -> Critique
        builder.add_edge("propose_action", "critique_action")
        
        # Critique -> Supervisor (Supervisor decides next step based on critique)
        builder.add_edge("critique_action", "supervisor")
        
        # Revise -> Critique (Critique the revised action)
        builder.add_edge("revise_action", "critique_action")
        
        # Compile the graph
        self.graph = builder.compile()
        # --- Visualize Graph ---
        try:
            print("--- LangGraph Mermaid Diagram ---")
            print(self.graph.get_graph().draw_mermaid())
            print("---------------------------------")
        except Exception as e:
            logger.warning(f"Could not generate graph diagram: {e}. Optional dependencies might be missing.")
        # ----------------------------------
        # print(self.graph.get_graph().print_ascii()) # Optional: print graph structure
        # --- End LangGraph Setup ---

        # Initialize Agent base class AFTER graph is ready
        super().__init__() # Call Agent's __init__

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode,
        )
        
        # Reset history (also done in reset method)
        self.action_history = []
        self.reflection_history = []

        # AgentLogger setup remains the same
        # Initialize the agent logger for Multion API logging
        # ... (rest of the logger setup remains the same) ...
        try:
            # Get API key from environment variables
            api_key = os.getenv("MULTION_API_KEY")
            if not api_key:
                logger.warning("MULTION_API_KEY not found in environment variables")
                self.agent_logger = None
            else:
                # Initialize the agent logger with a descriptive prompt
                initial_prompt = f"Reflexion Agent using {self.model_name} for browsergym interaction"
                # Ensure AgentLogger class is available (assuming it's defined elsewhere or imported)
                # from rl_training.agents.agent_logger_class import AgentLogger # Make sure this import works
                # self.agent_logger = AgentLogger(prompt=initial_prompt, api_key=api_key)
                # For now, comment out AgentLogger if it's not defined in this file
                self.agent_logger = None # Placeholder if AgentLogger class is unavailable
                if self.agent_logger:
                     print(f"Agent logger initialized with session ID: {self.agent_logger.SESSION_ID}")
                     self.agent_logger.log_step(
                         {"initialization": "Agent initialized", "model": self.model_name},
                         {"status": "ready", "chat_mode": self.chat_mode, "screenshot_enabled": self.use_screenshot}
                     )
        except NameError: # Catch if AgentLogger is not defined
             logger.warning("AgentLogger class not found. Logging disabled.")
             self.agent_logger = None
        except Exception as e:
            logger.error(f"Failed to initialize agent logger: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.agent_logger = None

        # Remove the old query_model function
        # self.query_model = query_model

    def _construct_prompt_messages(self, processed_obs: dict, current_step: Literal["propose", "critique", "revise"]) -> List[BaseMessage]:
        """Helper to construct the list of messages for the LLM, tailored to the current step."""
        messages = []

        # --- Base Information (System Prompt, Goal/Chat, Observations, History) ---
        # 1. System Prompt (Base instructions, Action Space)
        system_prompt_base = f"""\
# Instructions

You are a UI Assistant operating a web browser to help a user or achieve a goal.
Review the current state of the page, the user's request/goal, and your action/reflection history.
Your goal is to decide the single best next action to take through proposal and critique.

# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}
"""
        messages.append(SystemMessage(content=system_prompt_base))
        
        # 2. Goal / Chat History
        if self.chat_mode:
            messages.append(HumanMessage(content="# Chat History\n(Review messages to understand user intent)"))
            for msg in processed_obs["chat_messages"]:
                role_prefix = f"[{msg['role'].upper()}]"
                if msg["role"] == "user_image":
                    messages.append(HumanMessage(content=f"{role_prefix} (User sent an image)"))
                elif msg["role"] in ("user", "assistant", "infeasible"):
                    messages.append(HumanMessage(content=f"{role_prefix} {msg['message']}"))
                else:
                    logger.warning(f"Unexpected chat message role {repr(msg['role'])}")
                    messages.append(HumanMessage(content=f"[{msg['role'].upper()}] {msg['message']}"))
        else: # Goal-oriented mode
            goal_text = processed_obs.get("goal_object", "No goal specified.")
            messages.append(HumanMessage(content=f"# Goal\n\n{str(goal_text)}"))
            if isinstance(processed_obs.get("goal_object"), list):
                for item in processed_obs["goal_object"]:
                    if isinstance(item, dict) and 'type' in item and item['type'] == 'text' and 'text' in item:
                        messages.append(HumanMessage(content=item['text']))
                    elif isinstance(item, BaseMessage):
                        messages.append(item)

        # 3. Observation Details
        obs_content = []
        if self.use_axtree and processed_obs.get("axtree_txt"):
            obs_content.append(f"# Current page Accessibility Tree\n\n{processed_obs['axtree_txt']}")
        if self.use_html and processed_obs.get("pruned_html"):
            obs_content.append(f"# Current page DOM (pruned)\n\n{processed_obs['pruned_html']}")
        if self.use_screenshot and processed_obs.get("screenshot") is not None:
            try:
                img_url = image_to_jpg_base64_url(processed_obs["screenshot"])
                obs_content.append("# Current page Screenshot")
                messages.append(HumanMessage(content=[{"type": "text", "text": "\n".join(obs_content)}, {"type": "image_url", "image_url": {"url": img_url}}]))
                obs_content = [] 
            except Exception as e:
                logger.error(f"Failed to process screenshot for prompt: {e}")
                obs_content.append("# Current page Screenshot (Error processing image)")
        if obs_content:
            messages.append(HumanMessage(content="\n\n".join(obs_content)))

        # 4. Action & *Critique* History
        if self.action_history:
            history_content = ["# History (Past Actions and Critiques)"]
            for i, action in enumerate(self.action_history):
                critique_text = "No critique recorded."
                if i < len(self.reflection_history) and self.reflection_history[i]:
                    critique: Critique = self.reflection_history[i]
                    critique_text = f"  Critique: '{critique.critique}' (Sufficient: {critique.is_sufficient}, Missing: '{critique.missing}', Superfluous: '{critique.superfluous}')"
                history_content.append(f"- Action: {action}\n{critique_text}")
            if processed_obs.get("last_action_error"):
                history_content.append(f"\n# Error message from last action\n\n{processed_obs['last_action_error']}")
            messages.append(HumanMessage(content='\n'.join(history_content)))
        
        # --- Step-Specific Instructions --- 
        if current_step == "propose":
            messages.append(HumanMessage(content=f"# Task: Propose Action\n\nBased on the goal, observations, and history, propose the best single next action and your reasoning for it. Use the {ProposedAction.__name__} tool."))
        elif current_step == "critique":
            # Assume proposed_action is available in the state when constructing prompt for critique
             messages.append(HumanMessage(content=f"# Task: Critique Proposed Action\n\Critically evaluate the proposed action and reasoning provided in the previous step. Is it the best possible action? Is it safe? Does it directly address the goal and consider the history/errors? Provide detailed feedback. Use the {Critique.__name__} tool."))
        elif current_step == "revise":
             messages.append(HumanMessage(content=f"# Task: Revise Action\n\Based on the critique provided, revise your proposed action. Address the points raised in the critique. Use the {ProposedAction.__name__} tool to output the revised action and reasoning."))
        
        return messages

    # --- LangGraph Node Functions --- 

    def _supervisor_node(self, state: AgentState) -> dict:
        """Supervisor node to analyze state and decide routing."""
        logger.info("Invoking supervisor node...")
        # (Rule-based logic remains the same for now)
        # It reads proposed_action and critique from state
        # It returns {"supervisor_decision": ...}
        if state.get("proposed_action") is None:
             logger.info("Supervisor: No proposal yet, routing to propose.")
             return {"supervisor_decision": RoutingDecision(next_step="propose", reasoning="Initial step.")} 
        critique = state.get("critique")
        revision_attempts = state.get("revision_attempts", 0)
        max_revisions = 1
        if critique:
            if not critique.is_sufficient and revision_attempts < max_revisions:
                logger.info(f"Supervisor: Critique insufficient (Attempt {revision_attempts + 1}), routing to revise.")
                return {"supervisor_decision": RoutingDecision(next_step="revise", reasoning="Critique requires revision.")}
            else:
                if critique.is_sufficient:
                     logger.info("Supervisor: Critique sufficient, routing to final_answer.")
                     return {"supervisor_decision": RoutingDecision(next_step="final_answer", reasoning="Critique approved action.")}
                elif revision_attempts >= max_revisions:
                     logger.warning(f"Supervisor: Max revisions ({max_revisions}) reached, routing to final_answer with last proposal.")
                     return {"supervisor_decision": RoutingDecision(next_step="final_answer", reasoning="Max revisions reached.")}
        logger.warning("Supervisor: Fallback routing to propose (critique missing?).") # Added more context
        return {"supervisor_decision": RoutingDecision(next_step="propose", reasoning="Fallback decision - critique missing?")}

    def _propose_action_node(self, state: AgentState) -> dict:
        """Node that invokes the LLM to propose an action and reasoning."""
        logger.info("Invoking propose_action node...")
        propose_llm = self.llm.bind_tools(tools=[ProposedAction], tool_choice=ProposedAction.__name__)
        try:
            # Use INITIAL messages + Task instruction
            prompt_messages = state['initial_messages'] + [
                HumanMessage(content=f"# Task: Propose Action\n\nBased on the goal, observations, and history, propose the best single next action and your reasoning for it. Use the {ProposedAction.__name__} tool.")
            ]
            response = propose_llm.invoke(prompt_messages)

            if not response.tool_calls or response.tool_calls[0]['name'] != ProposedAction.__name__:
                logger.error(f"LLM did not return the expected {ProposedAction.__name__} tool call.")
                fallback_proposal = ProposedAction(action="report_infeasible('Proposal node failed')", reasoning="LLM failed format.")
                response = AIMessage(content="", tool_calls=[{"id": "tool_fallback_propose", "name": ProposedAction.__name__, "args": fallback_proposal.dict()}])
            
            parsed_proposal: ProposedAction = PydanticToolsParser(tools=[ProposedAction]).invoke(response)[0]
            logger.info(f"Proposed Action: {parsed_proposal.action}, Reasoning: {parsed_proposal.reasoning[:50]}...")
            # Return ONLY the new message and the proposal object
            return {"messages": [response], "proposed_action": parsed_proposal} 
        except Exception as e:
            logger.error(f"Error in propose_action_node: {e}")
            fallback_proposal = ProposedAction(action="report_infeasible('Error in proposal node')", reasoning=f"Exception: {e}")
            response = AIMessage(content="", tool_calls=[{"id": "tool_error_propose", "name": ProposedAction.__name__, "args": fallback_proposal.dict()}])
            return {"messages": [response], "proposed_action": fallback_proposal}

    def _critique_action_node(self, state: AgentState) -> dict:
        """Node that invokes the LLM to critique the proposed action."""
        logger.info("Invoking critique_action node...")
        proposed_action = state.get("proposed_action")
        if not proposed_action:
             logger.error("Critique node called without a proposed action in state.")
             fallback_critique = Critique(critique="No action proposed for critique.", is_sufficient=False, missing="Proposal step failed.", superfluous="N/A")
             return {"critique": fallback_critique} 

        critique_llm = self.llm.bind_tools(tools=[Critique], tool_choice=Critique.__name__)
        try:
            # Use INITIAL messages + Proposal info + Task instruction
            prompt_messages = state['initial_messages'] + [
                 HumanMessage(content=f"# Proposed Action for Critique\nAction: `{proposed_action.action}`\nReasoning: {proposed_action.reasoning}"),
                 HumanMessage(content=f"# Task: Critique Proposed Action\n\Critically evaluate the proposed action and reasoning. Use the {Critique.__name__} tool.")
             ]
            response = critique_llm.invoke(prompt_messages)

            if not response.tool_calls or response.tool_calls[0]['name'] != Critique.__name__:
                 logger.error(f"LLM did not return the expected {Critique.__name__} tool call.")
                 fallback_critique = Critique(critique="LLM failed to provide critique.", is_sufficient=True, missing="Critique format error.", superfluous="N/A") 
                 response = AIMessage(content="", tool_calls=[{"id": "tool_fallback_critique", "name": Critique.__name__, "args": fallback_critique.dict()}])

            parsed_critique: Critique = PydanticToolsParser(tools=[Critique]).invoke(response)[0]
            logger.info(f"Critique: {parsed_critique.critique[:50]}... | Sufficient: {parsed_critique.is_sufficient}")
            # Return ONLY the new message and the critique object
            return {"messages": [response], "critique": parsed_critique}
        except Exception as e:
             logger.error(f"Error in critique_action_node: {e}")
             fallback_critique = Critique(critique=f"Exception during critique: {e}", is_sufficient=True, missing="Critique step failed.", superfluous="N/A")
             response = AIMessage(content="", tool_calls=[{"id": "tool_error_critique", "name": Critique.__name__, "args": fallback_critique.dict()}])
             return {"messages": [response], "critique": fallback_critique}

    def _revise_action_node(self, state: AgentState) -> dict:
        """Node that invokes the LLM to revise the proposed action based on critique."""
        logger.info("Invoking revise_action node...")
        proposed_action = state.get("proposed_action")
        critique = state.get("critique")
        if not proposed_action or not critique:
             logger.error("Revise node called without proposed action or critique.")
             return {}

        revise_llm = self.llm.bind_tools(tools=[ProposedAction], tool_choice=ProposedAction.__name__)
        try:
            # Use INITIAL messages + Previous Proposal + Critique + Task instruction
            prompt_messages = state['initial_messages'] + [
                 HumanMessage(content=f"# Previous Proposed Action\nAction: `{proposed_action.action}`\nReasoning: {proposed_action.reasoning}"),
                 HumanMessage(content=f"# Critique Received\nCritique: {critique.critique}\nMissing: {critique.missing}\nSuperfluous: {critique.superfluous}"),
                 HumanMessage(content=f"# Task: Revise Action\n\nBased *specifically* on the critique provided, revise your proposed action and reasoning. Use the {ProposedAction.__name__} tool.")
            ]
            response = revise_llm.invoke(prompt_messages)
            
            if not response.tool_calls or response.tool_calls[0]['name'] != ProposedAction.__name__:
                 logger.error(f"LLM did not return the expected {ProposedAction.__name__} tool call during revision.")
                 # Return empty update? Or keep original proposal?
                 return {"messages": [response]} # Let add_messages handle the error message
            
            parsed_revised_proposal: ProposedAction = PydanticToolsParser(tools=[ProposedAction]).invoke(response)[0]
            logger.info(f"Revised Action: {parsed_revised_proposal.action}, Reasoning: {parsed_revised_proposal.reasoning[:50]}...")
            # Return updates: new message, revised proposal, incremented attempts
            return {
                "messages": [response], 
                "proposed_action": parsed_revised_proposal, 
                "revision_attempts": state.get("revision_attempts", 0) + 1
            }
        except Exception as e:
            logger.error(f"Error in revise_action_node: {e}")
            return {}

    # --- Conditional Edge Logic --- 

    def _route_from_supervisor(self, state: AgentState) -> Literal["propose", "revise", "force_infeasible", "final_answer"]:
        """Reads the supervisor decision from state and returns the routing target."""
        decision = state.get("supervisor_decision")
        if decision:
             # Handle force_infeasible and final_answer as END signals for the conditional edge
             if decision.next_step in ["force_infeasible", "final_answer"]:
                  return END
             return decision.next_step
        logger.warning("Supervisor decision not found in state, defaulting to propose.")
        return "propose" # Fallback routing

    def get_action(self, obs: dict) -> tuple[str, dict]:
        # 1. Preprocess observation (if needed, but harness usually does it)
        processed_obs = obs
        # 2. Construct base messages for the graph
        initial_messages = self._construct_base_messages_for_graph(processed_obs)

        # 3. Invoke the Graph with Supervisor
        graph_input = {
             "initial_messages": initial_messages, # Pass base messages
             "messages": [], # Start with empty running messages for this cycle
             "proposed_action": None, 
             "critique": None,       
             "supervisor_decision": None, # Start with no decision
             "revision_attempts": 0 
        }
        logger.info("Invoking Supervisor graph...")
        final_state = self.graph.invoke(graph_input)
        logger.info("Supervisor graph finished.")
        final_supervisor_decision = final_state.get("supervisor_decision")
        final_proposed_action = final_state.get("proposed_action") 
        final_critique = final_state.get("critique") 
        action_str = "report_infeasible(\"Error: Could not determine final action after supervisor review.\")"
        stored_critique = final_critique 
        if final_supervisor_decision:
             if final_supervisor_decision.next_step == "force_infeasible":
                  action_str = f"report_infeasible(\"Supervisor forced stop: {final_supervisor_decision.reasoning}\")"
                  logger.warning(f"Supervisor forced infeasible: {final_supervisor_decision.reasoning}")
             elif final_supervisor_decision.next_step == "final_answer" and final_proposed_action:
                  action_str = final_proposed_action.action
                  logger.info(f"Final Action Chosen (approved by supervisor): {action_str}")
             elif final_proposed_action:
                  logger.warning("Supervisor ended unexpectedly, using last proposed action.")
                  action_str = final_proposed_action.action
        elif final_proposed_action:
             logger.error("Supervisor decision missing, using last proposed action as fallback.")
             action_str = final_proposed_action.action
        else:
             logger.error("Supervisor decision and final proposal missing.")
        if not isinstance(stored_critique, Critique):
            stored_critique = Critique(critique="No valid critique generated or graph failed.", is_sufficient=False, missing="N/A", superfluous="N/A")
        self.action_history.append(action_str)
        self.reflection_history.append(stored_critique)
        if hasattr(self, 'agent_logger') and self.agent_logger is not None:
             try:
                 inputs = {
                    "prompt_summary": "Supervisor Graph Invoked", 
                    "observation": { 
                         "num_chat_messages": len(processed_obs.get("chat_messages", [])),
                         "has_screenshot": "screenshot" in processed_obs and processed_obs["screenshot"] is not None,
                         "has_axtree": "axtree_txt" in processed_obs and processed_obs["axtree_txt"] is not None,
                         "has_html": "pruned_html" in processed_obs and processed_obs["pruned_html"] is not None,
                         "goal": str(processed_obs.get("goal_object", ""))[:100] + "..." if processed_obs.get("goal_object") and len(str(processed_obs["goal_object"])) > 100 else str(processed_obs.get("goal_object", "")),
                         "last_action": processed_obs.get("last_action", ""),
                         "last_action_error": processed_obs.get("last_action_error", ""),
                    }
                 }
                 outputs = {
                     "action": action_str,
                     "action_type": action_str.split("(")[0] if "(" in action_str else "unknown",
                     "critique_missing": stored_critique.missing,
                     "critique_superfluous": stored_critique.superfluous,
                     "critique_sufficient": stored_critique.is_sufficient,
                     "critique_text": stored_critique.critique,
                     "supervisor_decision": final_supervisor_decision.next_step if final_supervisor_decision else "N/A"
                 }
                 action_type = action_str.split("(")[0] if "(" in action_str else "unknown"
                 action_args = action_str.split("(", 1)[1].rstrip(")") if "(" in action_str else ""
                 step_count = len(self.action_history)
                 supervisor_info = f" | Supervisor: {final_supervisor_decision.next_step}" if final_supervisor_decision else ""
                 print(f"Step {step_count}: {action_type} {action_args[:30]}{'...' if len(action_args) > 30 else ''}{supervisor_info}")
             except Exception as e:
                 logger.error(f"Failed to log step: {e}")
        return action_str, {}

    def _construct_base_messages_for_graph(self, processed_obs: dict) -> List[BaseMessage]:
        """Helper to construct the common part of messages list for graph nodes."""
        # This is essentially the logic from the old _construct_prompt_messages, 
        # minus the final step-specific instruction.
        messages = []
        # 1. System Prompt (Base instructions, Action Space)
        system_prompt_base = f"""\
# Instructions

You are a UI Assistant operating a web browser to help a user or achieve a goal.
Review the current state of the page, the user's request/goal, and your action/critique history.
Your goal is to decide the single best next action to take through proposal and critique.

# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}
"""
        messages.append(SystemMessage(content=system_prompt_base))
        
        # 2. Goal / Chat History
        if self.chat_mode:
            messages.append(HumanMessage(content="# Chat History\n(Review messages to understand user intent)"))
            for msg in processed_obs["chat_messages"]:
                role_prefix = f"[{msg['role'].upper()}]"
                if msg["role"] == "user_image":
                    messages.append(HumanMessage(content=f"{role_prefix} (User sent an image)"))
                elif msg["role"] in ("user", "assistant", "infeasible"):
                    messages.append(HumanMessage(content=f"{role_prefix} {msg['message']}"))
                else:
                    logger.warning(f"Unexpected chat message role {repr(msg['role'])}")
                    messages.append(HumanMessage(content=f"[{msg['role'].upper()}] {msg['message']}"))
        else: # Goal-oriented mode
            goal_text = processed_obs.get("goal_object", "No goal specified.")
            messages.append(HumanMessage(content=f"# Goal\n\n{str(goal_text)}"))
            if isinstance(processed_obs.get("goal_object"), list):
                for item in processed_obs["goal_object"]:
                    if isinstance(item, dict) and 'type' in item and item['type'] == 'text' and 'text' in item:
                        messages.append(HumanMessage(content=item['text']))
                    elif isinstance(item, BaseMessage):
                        messages.append(item)

        # 3. Observation Details
        obs_content = []
        if self.use_axtree and processed_obs.get("axtree_txt"):
            obs_content.append(f"# Current page Accessibility Tree\n\n{processed_obs['axtree_txt']}")
        if self.use_html and processed_obs.get("pruned_html"):
            obs_content.append(f"# Current page DOM (pruned)\n\n{processed_obs['pruned_html']}")
        if self.use_screenshot and processed_obs.get("screenshot") is not None:
            try:
                img_url = image_to_jpg_base64_url(processed_obs["screenshot"])
                obs_content.append("# Current page Screenshot")
                messages.append(HumanMessage(content=[{"type": "text", "text": "\n".join(obs_content)}, {"type": "image_url", "image_url": {"url": img_url}}]))
                obs_content = [] 
            except Exception as e:
                logger.error(f"Failed to process screenshot for prompt: {e}")
                obs_content.append("# Current page Screenshot (Error processing image)")
        if obs_content:
            messages.append(HumanMessage(content="\n\n".join(obs_content)))

        # 4. Action & Critique History
        if self.action_history:
            history_content = ["# History (Past Actions and Critiques)"]
            for i, action in enumerate(self.action_history):
                critique_text = "No critique recorded."
                if i < len(self.reflection_history) and self.reflection_history[i]:
                    critique: Critique = self.reflection_history[i]
                    critique_text = f"  Critique: '{critique.critique}' (Sufficient: {critique.is_sufficient}, Missing: '{critique.missing}', Superfluous: '{critique.superfluous}')"
                history_content.append(f"- Action: {action}\n{critique_text}")
            if processed_obs.get("last_action_error"):
                history_content.append(f"\n# Error message from last action\n\n{processed_obs['last_action_error']}")
            messages.append(HumanMessage(content='\n'.join(history_content)))
        
        return messages


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):

    agent_name: str = "ReflexionDemoAgent"  # Updated agent name
    model_name: str = "gpt-4o"     # Default model, can be changed
    chat_mode: bool = False        # Whether to enable chat mode
    demo_mode: str = "off"         # Visual effects mode (off, minimal, full)
    use_html: bool = False         # Whether to include HTML in observations
    use_axtree: bool = True        # Whether to include accessibility tree
    use_screenshot: bool = True    # Whether to include screenshots - useful for reflection
    # system_message_handling is no longer needed here

    def make_agent(self):
        """Create and return an instance of the Reflexion-enhanced DemoAgent."""
        return DemoAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            # system_message_handling removed
        )

# Update the run function to use the potentially modified AgentArgs
def run_demo_agent(model_name="gpt-4o", headless=True, leaderboard=False, run_id=None, task_name="webclones.omnizon-1"):    # Default headless=True for faster runs
    # Create the agent arguments with the specified parameters
    agent_args = DemoAgentArgs(
        model_name=model_name,
        chat_mode=False, # Defaulting to goal-oriented mode
        demo_mode="off",
        use_html=False, # Keep HTML off by default unless needed
        use_axtree=True,
        use_screenshot=True, # Enable screenshot by default for better context
        # system_message_handling removed
    )
    
    # Pass the agent arguments to the harness through the agisdk module
    # Note: The harness might still expect use_axtree/use_screenshot, ensure they match agent args
    harness = REAL.harness(
        # parallel=True, # Turn off parallelism for easier debugging initially
        # num_workers=1, # Use 1 worker with parallelism off
        agentargs=agent_args,
        # task_name="webclones.omnizon-1", # Use passed task_name
        task_type="omnizon",
        # task_type="omnizon", # Use passed task_name instead of hardcoded type
        headless=headless,          # Use passed headless value
        max_steps=25,               # Maximum steps per task
        use_axtree=agent_args.use_axtree, # Pass through from agent args
        use_screenshot=agent_args.use_screenshot, # Pass through from agent args
        leaderboard=leaderboard,    # Whether to submit to leaderboard
        run_id=run_id,              # Run ID for leaderboard submission
        use_cache=False
    )
    
    # Run the task
    logger.info(f"Running task '{task_name}' with agent '{agent_args.agent_name}'...")
    results = harness.run()
    
    # Print results summary
    logger.info("Task completed")
    logger.info(f"Results: {results}")
    
    return results


if __name__ == "__main__":
    # Example: Run the default Omnizon task
    # You can modify parameters here or via command-line args if added later
    results = run_demo_agent(task_name="webclones.omnizon-1", headless=True) 
    # Example with a different task and visible browser:
    # results = run_demo_agent(task_name="webclones.dashdish-1", headless=False) 