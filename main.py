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

# NEW IMPORTS for LangGraph and Pydantic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic.v1 import BaseModel, Field # <-- Use pydantic.v1 compatibility
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI # Ensure OpenAI is imported correctly

# Define Pydantic models for structured LLM output
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing in the proposed action or thought process.")
    superfluous: str = Field(description="Critique of what is superfluous or incorrect in the proposed action or thought process.")

class ActionDecision(BaseModel):
    """The final decision on the action to take, including self-reflection."""
    action: str = Field(description="The action string to execute in the environment (e.g., 'click(\"12\")').")
    reflection: Reflection = Field(description="Your reflection on the reasoning and chosen action.")


# Define LangGraph state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # We might add more state elements later if needed (e.g., original_input)


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

        # Define the Actor Node (generates action + reflection)
        builder.add_node("actor", self._actor_node)

        # Define edges (simple: start -> actor -> end)
        builder.add_edge(START, "actor")
        builder.add_edge("actor", END)

        # Compile the graph
        self.graph = builder.compile()
        # print(self.graph.get_graph().print_ascii())
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
        # self.query_model = query_model # Removed

    def _construct_prompt_messages(self, processed_obs: dict) -> List[BaseMessage]:
        """Helper to construct the list of messages for the LLM."""
        messages = []

        # 1. System Prompt (incorporating Reflexion instructions)
        system_prompt_text = f"""\
# Instructions

You are a UI Assistant operating a web browser to help a user or achieve a goal.
Review the current state of the page, the user's request/goal, your past actions, and crucially, your *past reflections* on those actions.
Think step-by-step to determine the best next action.
Critique your own reasoning process: identify missing considerations and potential flaws or superfluous steps in your plan *before* making a final decision.
Output your final action and your reflection using the required {ActionDecision.__name__} tool format.

# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}

# Reflection Guide

When reflecting, consider:
- Is the planned action directly helping achieve the goal?
- Are there simpler or more direct actions available?
- Did I miss any important elements on the page?
- Is the action based on correct understanding of the page state?
- Does this action address points raised in previous reflections?
"""
        messages.append(SystemMessage(content=system_prompt_text))

        # 2. Goal / Chat History
        if self.chat_mode:
            messages.append(HumanMessage(content="# Chat History\n(Review messages to understand user intent)"))
            for msg in processed_obs["chat_messages"]:
                 role_prefix = f"[{msg['role'].upper()}]"
                 if msg["role"] == "user_image":
                     # LangChain messages handle images differently, we might need adjustment
                     # For now, represent as text placeholder
                     messages.append(HumanMessage(content=f"{role_prefix} (User sent an image)"))
                 elif msg["role"] in ("user", "assistant", "infeasible"):
                     messages.append(HumanMessage(content=f"{role_prefix} {msg['message']}"))
                 else:
                     logger.warning(f"Unexpected chat message role {repr(msg['role'])}")
                     messages.append(HumanMessage(content=f"[{msg['role'].upper()}] {msg['message']}"))

        else: # Goal-oriented mode
            goal_text = processed_obs.get("goal_object", "No goal specified.")
            messages.append(HumanMessage(content=f"# Goal\n\n{goal_text}"))
            # Assuming goal_object is text or directly usable as message content
            if isinstance(goal_text, list): # Handle if goal_object is list of messages
                 messages.extend(goal_text)


        # 3. Observation Details (AXTree, HTML, Screenshot)
        obs_content = []
        if self.use_axtree and processed_obs.get("axtree_txt"):
            obs_content.append(f"# Current page Accessibility Tree\n\n{processed_obs['axtree_txt']}")
        if self.use_html and processed_obs.get("pruned_html"):
             obs_content.append(f"# Current page DOM (pruned)\n\n{processed_obs['pruned_html']}")
        
        # Handle Screenshot - LangChain expects image URLs or base64 in message content
        if self.use_screenshot and processed_obs.get("screenshot") is not None:
             try:
                 img_url = image_to_jpg_base64_url(processed_obs["screenshot"])
                 # Add text marker and the image message
                 obs_content.append("# Current page Screenshot")
                 messages.append(HumanMessage(content=[{"type": "text", "text": "\n".join(obs_content)}, {"type": "image_url", "image_url": {"url": img_url}}]))
                 obs_content = [] # Clear obs_content as it's now part of the image message
             except Exception as e:
                 logger.error(f"Failed to process screenshot for prompt: {e}")
                 obs_content.append("# Current page Screenshot (Error processing image)")
        
        # Add any remaining text observations
        if obs_content:
            messages.append(HumanMessage(content="\n\n".join(obs_content)))


        # 4. Action & Reflection History
        if self.action_history:
            history_content = ["# History (Past Actions and Reflections)"]
            for i, action in enumerate(self.action_history):
                reflection_text = "No reflection recorded."
                if i < len(self.reflection_history) and self.reflection_history[i]:
                     reflection = self.reflection_history[i]
                     reflection_text = f"  Reflection: Missing: '{reflection.missing}', Superfluous: '{reflection.superfluous}'"
                history_content.append(f"- Action: {action}\n{reflection_text}")
            
            if processed_obs.get("last_action_error"):
                history_content.append(f"\n# Error message from last action\n\n{processed_obs['last_action_error']}")
            
            messages.append(HumanMessage(content='\n'.join(history_content)))

        # 5. Final Instruction to Act
        messages.append(HumanMessage(content="# Next Action and Reflection\n\nReview all the information above. Think step-by-step, reflect on your reasoning, and then provide your chosen action and reflection using the required tool format."))
        
        return messages

    def _actor_node(self, state: AgentState) -> dict:
        """Node that invokes the LLM to generate action and reflection."""
        logger.info("Invoking actor node...")
        # Assumes the state['messages'] has been prepared correctly before calling the graph
        
        # Handle dummy key simulation if needed
        openai_api_key = os.getenv("OPENAI_API_KEY", "sk-dummy-key-for-testing")
        if openai_api_key.startswith("sk-dummy"):
             logger.warning("Using dummy API key - simulating LLM response.")
             # Simulate a response conforming to the ActionDecision tool
             simulated_response = AIMessage(
                 content="", 
                 tool_calls=[{
                     "id": "tool_dummy_123",
                     "name": ActionDecision.__name__,
                     "args": {
                         "action": "click(\"1\")",
                         "reflection": {"missing": "None", "superfluous": "None"}
                     }
                 }]
             )
             return {"messages": [simulated_response]}

        try:
            # Invoke the LLM with the prepared messages and bound tool
            response = self.llm_with_tool.invoke(state["messages"])
            # We expect the response to be an AIMessage with a tool_call
            if not response.tool_calls or response.tool_calls[0]['name'] != ActionDecision.__name__:
                 # Fallback or error handling if the LLM didn't use the tool
                 logger.error(f"LLM did not return the expected {ActionDecision.__name__} tool call. Response: {response}")
                 # Attempt to coerce or return an error action
                 fallback_action = "send_msg_to_user(\"Internal error: Failed to decide action.\")"
                 fallback_reflection = Reflection(missing="LLM failed to use the required format.", superfluous="None")
                 response = AIMessage(content="", tool_calls=[{"id": "tool_fallback_123", "name": ActionDecision.__name__, "args": {"action": fallback_action, "reflection": fallback_reflection.dict()}}])
            
            return {"messages": [response]} # Append the LLM's response (with tool call)
        except Exception as e:
            logger.error(f"Error calling LLM in actor node: {e}")
            # Return a safe fallback action
            error_action = "send_msg_to_user(\"I encountered an error processing my decision.\")"
            error_reflection = Reflection(missing="Error occurred during LLM call.", superfluous="None")
            error_response = AIMessage(content="", tool_calls=[{"id": "tool_error_123", "name": ActionDecision.__name__, "args": {"action": error_action, "reflection": error_reflection.dict()}}])
            return {"messages": [error_response]}


    def get_action(self, obs: dict) -> tuple[str, dict]:
        # 1. Preprocess observation (already done by harness)
        # processed_obs = self.obs_preprocessor(obs) # obs is already processed by harness
        processed_obs = obs # Use the observation directly passed by the harness

        # 2. Construct prompt messages
        prompt_messages = self._construct_prompt_messages(processed_obs)

        # Log the prompt text (optional, can be verbose)
        # full_prompt_txt = "\n".join([str(m.content) for m in prompt_messages if isinstance(m.content, str)])
        # logger.info(f"--- Sending Prompt to LLM ---\n{full_prompt_txt[:1000]}...\n--- End Prompt ---")

        # 3. Invoke the LangGraph graph
        graph_input = {"messages": prompt_messages}
        final_state = self.graph.invoke(graph_input)

        # 4. Parse the result from the final state
        last_message = final_state["messages"][-1]
        action_str = "send_msg_to_user(\"Error: Could not determine action.\")" # Default fallback
        current_reflection = Reflection(missing="Result parsing failed.", superfluous="") # Default fallback

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if tool_call['name'] == ActionDecision.__name__:
                try:
                    parsed_result: ActionDecision = self.parser.invoke(last_message)[0] # Parse Pydantic model
                    action_str = parsed_result.action
                    current_reflection = parsed_result.reflection
                    logger.info(f"Action chosen: {action_str}")
                    logger.info(f"Reflection: Missing='{current_reflection.missing}', Superfluous='{current_reflection.superfluous}'")
                except Exception as e:
                    logger.error(f"Failed to parse ActionDecision from LLM response: {e}. Tool call args: {tool_call.get('args')}")
            else:
                 logger.error(f"Unexpected tool call in final message: {tool_call['name']}")
        else:
             logger.error(f"Unexpected final message type or content: {last_message}")


        # 5. Store action and reflection history (before returning action)
        self.action_history.append(action_str)
        self.reflection_history.append(current_reflection)

        # 6. Log step via AgentLogger (if enabled)
        if hasattr(self, 'agent_logger') and self.agent_logger is not None:
            try:
                 # (Keep the existing logging logic, adapting inputs/outputs as needed)
                 inputs = {
                     "prompt_summary": "Prompt constructed with history and reflection.", # Simplify logging
                     "observation": { # Keep observation summary
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
                     "reflection_missing": current_reflection.missing,
                     "reflection_superfluous": current_reflection.superfluous,
                 }
                 # self.agent_logger.log_step(inputs, outputs) # Uncomment if AgentLogger is available

                 action_type = action_str.split("(")[0] if "(" in action_str else "unknown"
                 action_args = action_str.split("(", 1)[1].rstrip(")") if "(" in action_str else ""
                 # step_count = self.agent_logger.step_count if self.agent_logger else len(self.action_history) # Get step count
                 step_count = len(self.action_history) # Use action history length as step count
                 print(f"Step {step_count}: {action_type} {action_args[:30]}{'...' if len(action_args) > 30 else ''} | Reflection: {current_reflection.missing[:30]}...")

            except Exception as e:
                logger.error(f"Failed to log step to Multion API: {e}")

        # 7. Return the action string required by the AGI SDK harness
        return action_str, {} # Return empty dict as second element


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