import base64
import io
import logging
import os
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
from openai import OpenAI

# Assuming agisdk is installed and accessible
from agisdk import REAL
from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
from agisdk.REAL.browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from agisdk.REAL.demo_agent.run_demo import str2bool # For args if needed later
import dataclasses


logger = logging.getLogger(__name__)

# --- Reusable Helper Functions ---
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

# --- Agent States ---
class AgentState(Enum):
    NEEDS_PLAN = auto()
    EXECUTING_PLAN = auto()
    PLAN_FAILED = auto() # State if plan execution encounters critical error
    # Add more states as needed (e.g., CRITIQUING_ACTION)

# --- Base Sub-Agent Logic ---
class BaseSubAgent:
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def _query_model(self, system_prompt: str, user_prompt: List[Dict[str, any]]) -> str:
        # Basic wrapper for OpenAI call - can be enhanced with error handling, retries
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying OpenAI model ({self.__class__.__name__}): {str(e)}")
            # Return a structured error or raise?
            return f"ERROR: Could not query model: {str(e)}"

# --- Sub-Agents Implementation ---

class PlannerAgent(BaseSubAgent):
    def get_plan(self, obs: dict) -> List[str]:
        """Generates a step-by-step plan based on the goal and current observation."""
        
        system_prompt = ( "You are a planning agent. Your task is to analyze the user's goal and the current web page state "
                          "to create a concise, step-by-step plan to achieve the goal. Focus on breaking down the task into "
                          "logical, sequential browser actions. Output ONLY the numbered plan steps, each on a new line."
                         )

        # Prepare user prompt content (customize based on available obs keys)
        user_prompt_content = []
        user_prompt_content.append({"type": "text", "text": f"# Goal\n\n{obs['goal_object']}"})
        
        # Include relevant observation data (e.g., AXTree, HTML, Screenshot)
        # Choose based on what gives the planner the best context
        if obs.get("axtree_txt"):
             user_prompt_content.append({"type": "text", "text": f"\n# Current Page Accessibility Tree\n\n{obs['axtree_txt']}"})
        # Add HTML or Screenshot if configured/needed
        # if obs.get("pruned_html"):
        #     user_prompt_content.append({"type": "text", "text": f"\n# Current Page DOM\n\n{obs['pruned_html']}"})
        # if obs.get("screenshot"):
        #     user_prompt_content.append({"type": "image_url", "image_url": {"url": image_to_jpg_base64_url(obs["screenshot"]), "detail": "auto"}})

        user_prompt_content.append({
            "type": "text", 
            "text": ("\n# Plan\n\nGenerate the numbered plan steps. Example Output:\n"
                     "1. Click the login button.\n"
                     "2. Fill the username field.\n"
                     "3. Fill the password field.\n"
                     "4. Click submit.")
        })

        raw_plan = self._query_model(system_prompt, user_prompt_content)

        if raw_plan.startswith("ERROR:"):
            logger.error(f"Planner failed to generate plan: {raw_plan}")
            return [] # Return empty plan on error

        # Parse the raw plan (simple newline split, remove numbering)
        plan_steps = []
        for line in raw_plan.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # Attempt to remove common numbering prefixes (e.g., "1.", "- ")
            parts = line.split('.', 1)
            if len(parts) == 2 and parts[0].isdigit():
                step = parts[1].strip()
            elif line.startswith('- '):
                step = line[2:].strip()
            else:
                step = line
            
            if step: # Avoid adding empty steps
                plan_steps.append(step)
        
        logger.info(f"Generated Plan: {plan_steps}")
        return plan_steps

class ActorAgent(BaseSubAgent):
    def propose_action(self, obs: dict, current_plan_step: str, action_set_description: str) -> str:
        """Generates a specific action string based on the current plan step and observation."""

        system_prompt = ( "You are an acting agent. Your task is to execute the current step of a high-level plan "
                          "by generating the precise action command for a web browser environment. "
                          "Use the provided observation (Accessibility Tree, DOM, Screenshot) to find the correct elements and parameters (like bids) for the action. "
                          "Pay close attention to any previous error messages for the last action attempt."
                        )

        user_prompt_content = []
        user_prompt_content.append({"type": "text", "text": f"# Goal\n\n{obs['goal_object']}"})
        user_prompt_content.append({"type": "text", "text": f"\n# Current Plan Step to Execute\n\n{current_plan_step}"})

        # Include relevant observation data
        if obs.get("axtree_txt"):
             user_prompt_content.append({"type": "text", "text": f"\n# Current Page Accessibility Tree\n\n{obs['axtree_txt']}"})
        # if obs.get("pruned_html"):
        #     user_prompt_content.append({"type": "text", "text": f"\n# Current Page DOM\n\n{obs['pruned_html']}"})
        # if obs.get("screenshot"):
        #     user_prompt_content.append({"type": "image_url", "image_url": {"url": image_to_jpg_base64_url(obs["screenshot"]), "detail": "auto"}})

        # Include Action Space Description
        user_prompt_content.append({"type": "text", "text": f"\n# Available Actions\n\n{action_set_description}"})
        
        # Include last error if present
        if obs.get("last_action_error"):
            user_prompt_content.append({
                "type": "text",
                "text": f"""
# Error message from last attempt at this step

{obs["last_action_error"]}

Analyze this error and try to achieve the plan step (`{current_plan_step}`) successfully now.
"""
            })

        # Ask for the single action
        user_prompt_content.append({
            "type": "text",
            "text": ("\n# Next Action Command\n\n"
                     "Based on the plan step and the current page state, "
                     "output ONLY the single, precise command to execute next, enclosed in markdown code fences. "
                     "Example: ```click(\"12\")```")
        })

        raw_action_response = self._query_model(system_prompt, user_prompt_content)

        if raw_action_response.startswith("ERROR:"):
            logger.error(f"Actor failed to generate action: {raw_action_response}")
            # Maybe return a specific error action? For now, use send_msg_to_user
            return 'send_msg_to_user("Internal error: Actor failed to generate action.")'

        # --- Action Extraction Logic ---
        # Attempt to extract action from markdown code fences ```action(...)```
        import re
        match = re.search(r'```(.*?)```', raw_action_response, re.DOTALL)
        if match:
            action_str = match.group(1).strip()
            # Further clean the action string if necessary (e.g., remove leading/trailing quotes if LLM adds them)
            action_str = action_str.strip('"`') 
            logger.info(f"Actor proposed action: {action_str}")
            return action_str
        else:
            # Fallback: If no fences, maybe the LLM just output the action? (Less ideal)
            # Basic check if it looks like a function call
            potential_action = raw_action_response.strip()
            if re.match(r'^[a-zA-Z_]+\(.*\)$', potential_action):
                 logger.warning(f"Actor output action without fences: {potential_action}. Using it directly.")
                 return potential_action
            else:
                logger.error(f"Actor failed to extract action from response: {raw_action_response}")
                # Fallback action if extraction fails
                return 'send_msg_to_user("Internal error: Actor failed to parse action response.")'

class CriticAgent(BaseSubAgent): # Optional
    def evaluate_action(self, obs: dict, proposed_action: str) -> bool:
        # Implementation needed: Create prompt, call LLM, return critique (e.g., True=OK, False=Bad)
        pass

# --- Orchestrator Agent Implementation ---

class OrchestratorAgent(REAL.Agent):
    def __init__(self, args: 'OrchestratorAgentArgs'):
        super().__init__()
        self.args = args
        self.state = AgentState.NEEDS_PLAN
        self.current_plan: List[str] = []
        self.current_plan_step_index: int = 0

        # Initialize OpenAI Client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found, using dummy key.")
            openai_api_key = "sk-dummy"
        self.client = OpenAI(api_key=openai_api_key)

        # Initialize Sub-Agents
        self.planner = PlannerAgent(self.client, args.model_name) # Use same model for now
        self.actor = ActorAgent(self.client, args.model_name)     # Use same model for now
        # self.critic = CriticAgent(self.client, args.critic_model_name) # If using critic

        # Initialize Action Set for Actor prompts
        # TODO: Pass relevant args like demo_mode from OrchestratorAgentArgs if needed
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas"],
            strict=False,
            multiaction=False,
            # demo_mode=args.demo_mode 
        )
        self.action_set_description = self.action_set.describe(with_long_description=False, with_examples=True)


    def reset(self):
        super().reset()
        logger.info("Resetting OrchestratorAgent state.")
        self.state = AgentState.NEEDS_PLAN
        self.current_plan = []
        self.current_plan_step_index = 0

    def get_action(self, obs: dict) -> Tuple[str, Dict]:
        logger.info(f"Current state: {self.state}, Plan step: {self.current_plan_step_index}/{len(self.current_plan)}")
        
        action_str = 'send_msg_to_user("Orchestrator internal error.")' # Default error action
        metadata = {}

        try:
            # --- State Machine Logic ---
            if self.state == AgentState.PLAN_FAILED:
                logger.warning("Agent is in PLAN_FAILED state.")
                return 'send_msg_to_user("Planning failed. Cannot proceed.")', metadata

            # Handle planning if needed
            if self.state == AgentState.NEEDS_PLAN:
                logger.info("Generating a new plan...")
                self.current_plan = self.planner.get_plan(obs)
                if self.current_plan:
                    self.current_plan_step_index = 0
                    self.state = AgentState.EXECUTING_PLAN
                    logger.info(f"Plan generated successfully ({len(self.current_plan)} steps). Starting execution.")
                    # Proceed to execute the first step immediately
                else:
                    logger.error("Planner failed to return a plan.")
                    self.state = AgentState.PLAN_FAILED
                    return 'send_msg_to_user("Planning failed. Cannot generate a plan.")', metadata
            
            # Execute the current plan step
            if self.state == AgentState.EXECUTING_PLAN:
                if self.current_plan_step_index >= len(self.current_plan):
                    logger.info("Plan execution complete.")
                    return 'send_msg_to_user("Plan completed successfully.")', metadata # Or None? Check SDK behavior

                # Determine if we are retrying the step due to a previous error
                is_retrying = bool(obs.get('last_action_error'))
                if is_retrying:
                     logger.warning(f"Retrying plan step {self.current_plan_step_index} due to previous error.")
                
                # Get the current step text
                current_step_text = self.current_plan[self.current_plan_step_index]
                logger.info(f"Executing plan step {self.current_plan_step_index}: '{current_step_text}'")
                
                # Get action proposal from Actor
                action_str = self.actor.propose_action(obs, current_step_text, self.action_set_description)
                metadata = {"plan_step_index": self.current_plan_step_index, "plan_step_text": current_step_text}

                # --- Plan Advancement Logic ---
                # Advance the plan *only if* the previous action (related to the previous index) was successful.
                # We check the error from the *current* observation, which reflects the result of the *last* action taken.
                if not is_retrying and not action_str.startswith("ERROR:"): # If last action succeeded and Actor didn't fail
                    # Check if actor returned a failure message itself
                    if not action_str.startswith('send_msg_to_user("Internal error:'):
                       self.current_plan_step_index += 1
                       logger.info(f"Advancing to plan step {self.current_plan_step_index}")
                    else:
                        logger.warning("Actor returned an internal error message, not advancing plan.")
                elif is_retrying:
                    logger.info("Not advancing plan index because this was a retry.")
                else: # Actor failed internally
                     logger.error(f"Actor failed to propose a valid action for step {self.current_plan_step_index}. Not advancing plan.")
                    # State remains EXECUTING_PLAN, will retry on next call

            else:
                logger.error(f"Orchestrator entered unknown state: {self.state}")
        
        except Exception as e:
            logger.error(f"Exception in OrchestratorAgent.get_action: {e}", exc_info=True)
            # Optionally change state to PLAN_FAILED or another error state?

        return action_str, metadata

# --- Orchestrator Agent Arguments ---

@dataclasses.dataclass
class OrchestratorAgentArgs(REAL.AbstractAgentArgs):
    agent_name: str = "OrchestratorAgent"
    model_name: str = "gpt-4o" 
    # Configuration for sub-agents (can use different models if needed)
    # planner_model_name: str = "gpt-4o"
    # actor_model_name: str = "gpt-4o"
    # critic_model_name: str = "gpt-4o"

    # Observation config (pass these down if sub-agents need them)
    # use_screenshot: bool = False
    # use_axtree: bool = True 
    # use_html: bool = False

    # ActionSet config
    # demo_mode: str = "off"

    def make_agent(self) -> OrchestratorAgent:
        logger.info(f"Creating OrchestratorAgent with args: {self}")
        return OrchestratorAgent(self) 