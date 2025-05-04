import logging
import re
from typing import Dict, Optional, List
from openai import OpenAI
from .common import BaseSubAgent, image_to_jpg_base64_url 

logger = logging.getLogger(__name__)

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
                "text": f"""\n# Error message from last attempt at this step

{obs["last_action_error"]}

Analyze this error and try to achieve the plan step (`{current_plan_step}`) successfully now.
"""
            })
        
        # Include last critique if present (from rule-based critic)
        if obs.get('last_action_critique'):
             user_prompt_content.append({
                "type": "text",
                "text": f"""\n# Critique of your last proposed action for this step\n\n{obs['last_action_critique']}\n\nAddress this critique in your next action proposal for the plan step (`{current_plan_step}`).\n"""
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