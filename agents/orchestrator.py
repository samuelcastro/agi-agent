import dataclasses
import logging
import os
import json
import re
from typing import List, Tuple, Dict, Optional

from openai import OpenAI
from agisdk import REAL
from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet

# Import sub-agents and common components
from .common import AgentState, BaseSubAgent
from .planner import PlannerAgent
from .actor import ActorAgent
from .critic import CriticAgent

logger = logging.getLogger(__name__)


# --- Orchestrator Agent Arguments --- (Moved here from bottom)

@dataclasses.dataclass
class OrchestratorAgentArgs(REAL.AbstractAgentArgs):
    agent_name: str = "OrchestratorAgent"
    model_name: str = "gpt-4o" 
    # Configuration for sub-agents (can use different models if needed)
    # planner_model_name: str = "gpt-4o"
    # actor_model_name: str = "gpt-4o"
    # critic_model_name: str = "gpt-4o"

    # Observation config (pass these down if sub-agents need them)
    use_screenshot: bool = False
    # use_axtree: bool = True 
    # use_html: bool = False

    # ActionSet config
    # demo_mode: str = "off"

    use_critic: bool = True # Add option to enable/disable critic

    def make_agent(self) -> 'OrchestratorAgent': # Forward reference
        logger.info(f"Creating OrchestratorAgent with args: {self}")
        return OrchestratorAgent(self) 

# --- Orchestrator Agent Implementation ---

class OrchestratorAgent(REAL.Agent):
    def __init__(self, args: OrchestratorAgentArgs):
        super().__init__()
        self.args = args
        self.state = AgentState.NEEDS_PLAN
        self.current_plan: List[str] = []
        self.current_plan_step_index: int = 0
        self.last_critique: str = "" # Store critique for Actor retry

        # Initialize OpenAI Client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found, using dummy key.")
            openai_api_key = "sk-dummy"
        self.client = OpenAI(api_key=openai_api_key)

        # Initialize Sub-Agents
        self.planner = PlannerAgent(self.client, args.model_name)
        self.actor = ActorAgent(self.client, args.model_name)
        self.critic = CriticAgent(self.client, args.model_name)

        # Initialize Action Set for Actor prompts
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas"], # Added fill, select_option if they exist
            strict=False,
            multiaction=False,
            # demo_mode=args.demo_mode # demo_mode not defined in args yet
        )
        self.action_set_description = self.action_set.describe(with_long_description=False, with_examples=True)


    def reset(self):
        super().reset()
        logger.info("Resetting OrchestratorAgent state.")
        self.state = AgentState.NEEDS_PLAN
        self.current_plan = []
        self.current_plan_step_index = 0
        self.last_critique = ""

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
                # TODO: Add context about previous failure if replanning?
                # obs['replanning_context'] = f"Replanning after previous attempt failed at step X..."
                self.current_plan = self.planner.get_plan(obs)
                if self.current_plan:
                    self.current_plan_step_index = 0
                    self.state = AgentState.EXECUTING_PLAN
                    self.last_critique = "" # Clear any critique from previous plan
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
                 
                 # --- Actor-Critic Interaction Loop (Simplified: 1 attempt + critique) ---
                 # Add last critique to observation if retrying after critique
                 if self.last_critique:
                     obs['last_action_critique'] = self.last_critique # Add critique info
 
                 # Get action proposal from Actor
                 proposed_action_str = self.actor.propose_action(obs, current_step_text, self.action_set_description)
                 
                 # Reset critique after Actor uses it
                 self.last_critique = "" 
                 obs.pop('last_action_critique', None) # Clean up obs
 
                 # Validate with Critic (if enabled)
                 is_valid_action = True # Assume valid if critic is disabled
                 critique = "Critic disabled."
                 if self.args.use_critic:
                     is_valid_action, critique = self.critic.evaluate_action(obs, current_step_text, proposed_action_str)
 
                 action_str = proposed_action_str 
                 metadata = {"plan_step_index": self.current_plan_step_index, "plan_step_text": current_step_text}
 
                 # If Critic rejects, store critique and prepare for retry on next step
                 if not is_valid_action:
                     logger.warning(f"Critic rejected action: {critique}")
                     self.last_critique = critique
                     metadata["critique_result"] = f"Rejected: {critique}"
                     # Still return the proposed (invalid) action for the env to potentially error on
                     # Plan advancement is blocked below.
                 else:
                     # Action approved by critic or critic disabled
                     metadata["critique_result"] = "Approved" if self.args.use_critic else "Approval N/A (Critic Disabled)"
                     
                     # --- Plan Advancement Logic --- 
                     # Advance plan *only if* the critic approved (or is disabled)
                     # AND the last *execution* attempt was successful (or this is a valid retry)
                     if not is_retrying: # Last execution succeeded
                          if not action_str.startswith('send_msg_to_user("Internal error:'):
                              self.current_plan_step_index += 1
                              logger.info(f"Advancing to plan step {self.current_plan_step_index} after successful execution and critique approval.")
                          else:
                             logger.warning("Actor returned an internal error message, not advancing plan.")
                     else: # Last execution failed (we are retrying)
                         # Since the critic *approved* this new action (or is disabled), 
                         # we assume this retry is valid and advance the plan.
                         if not action_str.startswith('send_msg_to_user("Internal error:'):
                             self.current_plan_step_index += 1
                             logger.info(f"Advancing to plan step {self.current_plan_step_index} after successful retry and critique approval.")
                         else:
                              logger.warning("Actor returned an internal error message on retry, not advancing plan.")
 
            else:
                logger.error(f"Orchestrator entered unknown state: {self.state}")
        
        except Exception as e:
            logger.error(f"Exception in OrchestratorAgent.get_action: {e}", exc_info=True)
            # Optionally change state to PLAN_FAILED or another error state?

        return action_str, metadata 