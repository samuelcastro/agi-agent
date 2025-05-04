import logging
from typing import List, Dict, Optional
from openai import OpenAI
from .common import BaseSubAgent, image_to_jpg_base64_url

logger = logging.getLogger(__name__)


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