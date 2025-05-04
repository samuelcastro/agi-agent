import logging
from typing import Tuple, Optional, Dict, List
from openai import OpenAI
from .common import BaseSubAgent

logger = logging.getLogger(__name__)

class CriticAgent(BaseSubAgent):
    def __init__(self, client: OpenAI, model_name: str):
        # LLM is now needed, call the parent initializer
        super().__init__(client, model_name) 

    def evaluate_action(self, obs: dict, current_plan_step: str, proposed_action: str) -> Tuple[bool, str]:
        """Evaluates a proposed action using an LLM. Returns (is_valid, critique_message)."""
        logger.info(f"Critic evaluating action: {proposed_action} for step: {current_plan_step}")

        system_prompt = (
            "You are a meticulous critic agent. Your task is to evaluate if a proposed browser action "
            "is valid and appropriate given the current web page state (AXTree) and the specific plan step "
            "it is supposed to achieve. Focus on validity (Does the element likely exist? Is the action type suitable?) "
            "and relevance (Does this action directly help achieve the current plan step?). "
            "Consider any previous error messages." 
        )

        user_prompt_content = []
        user_prompt_content.append({"type": "text", "text": f"# Goal\n{obs.get('goal_object', 'N/A')}"})
        user_prompt_content.append({"type": "text", "text": f"\n# Current Plan Step\n{current_plan_step}"})
        
        if obs.get("axtree_txt"):
            user_prompt_content.append({"type": "text", "text": f"\n# Current Page Accessibility Tree\n{obs['axtree_txt']}"})
        # Optionally add HTML or screenshot if needed

        user_prompt_content.append({"type": "text", "text": f"\n# Actor's Proposed Action\n```{proposed_action}```"})

        if obs.get("last_action_error"):
            user_prompt_content.append({"type": "text", "text": f"\n# Last Action Error (for context)\n{obs['last_action_error']}"})

        user_prompt_content.append({
            "type": "text", 
            "text": ("\n# Evaluation\nIs the proposed action valid and relevant for the current plan step, given the page state and potential errors? "
                     "Respond ONLY with the word 'Valid.' or 'Invalid.' followed by a concise reason. "
                     "Example Valid: Valid. The click action targets an existing button relevant to the step. "
                     "Example Invalid: Invalid. The proposed bid does not exist in the AXTree. "
                     "Example Invalid: Invalid. select_option cannot be used on a div element.")
        })

        llm_response = self._query_model(system_prompt, user_prompt_content)

        if llm_response.startswith("ERROR:"):
            logger.error(f"Critic LLM query failed: {llm_response}")
            # Default to invalid if LLM fails
            return False, f"Critic LLM query failed: {llm_response}"

        # Parse response
        response_lower = llm_response.lower().strip()
        if response_lower.startswith("valid"):
            logger.info(f"Critic approves action. Reason: {llm_response}")
            # Return the full response as the critique message for context
            return True, llm_response 
        else:
            # Assume invalid otherwise
            logger.warning(f"Critic rejects action. Reason: {llm_response}")
            return False, llm_response 