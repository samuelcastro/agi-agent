# Agent Optimization Plan

**Goal:** Optimize the `DemoAgent` in `main.py` using reflection, self-critique, and insights from the AGI paper and `agisdk` documentation to improve its task success rate and robustness on web-based tasks.

**Analysis Summary:**

- **Current Agent (`main.py`):** Uses `agisdk`, takes observations (page state, history, errors), prompts GPT-4o for actions. Maintains action history.
- **`agisdk`:** Provides framework, environment (`BrowserGym`), base `Agent` class, observation/action structure. Supports custom agents.
- **AGI Paper:** Advocates for guided search (MCTS), self-critique, and learning from trajectories (DPO) over simple imitation learning for web agents. Shows significant performance gains.
- **Example Code:** Implements a multi-agent structure (Planner, BrowserNav, Actor, Critic) for task decomposition.

**Proposed Optimization Tasks:**

1.  **Enhance Prompt for Self-Critique & Error Correction:**

    - Modify the prompt structure in `DemoAgent.get_action`.
    - Explicitly instruct the agent to analyze the `last_action` and `last_action_error`.
    - If `last_action_error` is present, require the agent to:
      - Hypothesize the cause of the error.
      - Explain how the next action avoids repeating the mistake.
    - _Target Code:_ `main.py` (`get_action` method, prompt construction logic).

2.  **Add Explicit Reflection Step:**

    - Introduce a new section in the prompt (e.g., `# Reflection`).
    - Prompt the agent to briefly:
      - Summarize its current progress towards the goal.
      - Assess if the previous actions were effective.
      - Re-evaluate its high-level plan if necessary.
    - _Target Code:_ `main.py` (`get_action` method, prompt construction logic).

3.  **Refine Action History Usage:**

    - Modify the prompt instructions related to `# History of past actions`.
    - Encourage the agent to explicitly state how the history informs the _next_ action, rather than just listing it.
    - _Target Code:_ `main.py` (`get_action` method, prompt construction logic).

4.  **(Optional) Simple Planning Prompt:**
    - Consider adding a request for a brief plan (next 1-3 steps) within the agent's reasoning process before it outputs the final action string. This encourages more deliberate thinking.
    - _Target Code:_ `main.py` (`get_action` method, prompt construction logic).

**Implementation:**

- Apply the selected prompt modifications sequentially to the `get_action` method in `main.py`.
- Test changes iteratively.

**Evaluation (Future):**

- Define specific benchmark tasks from the `agisdk` (e.g., `webclones.omnizon-*`, `webclones.opendining-*`).
- Measure success rate, number of steps, and error frequency before and after optimization.
- Compare results against baseline performance.
