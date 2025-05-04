import base64
import io
import logging
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
from openai import OpenAI

logger = logging.getLogger(__name__)

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