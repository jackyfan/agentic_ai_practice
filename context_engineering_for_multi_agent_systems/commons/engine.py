from helpers import call_llm_robust
import json


def planner(goal, capabilities):
    """
     Analyzes the goal and generates a structured Execution Plan using the LLM.
     """
    print("[引擎:规划器] 分析目标并生成执行计划...")

    system_prompt = f"""
    You are the strategic core of the Context Engine. Analyze the user's high-level 
    goal and create a structured Execution Plan using the available agents.
    --- AVAILABLE CAPABILITIES ---
    {capabilities}
    --- END CAPABILITIES ---
    INSTRUCTIONS:
    1. The plan MUST be a JSON list of objects, where each object is a "step".
    2. You MUST use Context Chaining. If a step requires input from a previous step, 
    reference it using the syntax $$STEP_X_OUTPUT$$.
    3. Be strategic. Break down complex goals (like sequential rewriting) into 
    distinct steps. Use the correct input keys ('facts' vs 'previous_content') for the 
    Writer agent.
    EXAMPLE GOAL: "Write a suspenseful story about Apollo 11."
    EXAMPLE PLAN (JSON LIST):
    [
     {{"step": 1, "agent": "Librarian", "input": {{"intent_query": "suspenseful 
    narrative blueprint"}}}},
     {{"step": 2, "agent": "Researcher", "input": {{"topic_query": "Apollo 11 
    landing details"}}}},
     {{"step": 3, "agent": "Writer", "input": {{"blueprint": "$$STEP_1_OUTPUT$$", 
    "facts": "$$STEP_2_OUTPUT$$"}}}}
    ]
    EXAMPLE GOAL: "Write a technical report on Juno, then rewrite it casually."
    EXAMPLE PLAN (JSON LIST):
    [
     {{"step": 1, "agent": "Librarian", "input": {{"intent_query": "technical 
    report structure"}}}},
     {{"step": 2, "agent": "Researcher", "input": {{"topic_query": "Juno mission 
    technology"}}}},
     {{"step": 3, "agent": "Writer", "input": {{"blueprint": "$$STEP_1_OUTPUT$$", 
    "facts": "$$STEP_2_OUTPUT$$"}}}},
     {{"step": 4, "agent": "Librarian", "input": {{"intent_query": "casual summary 
    style"}}}},
     {{"step": 5, "agent": "Writer", "input": {{"blueprint": "$$STEP_4_OUTPUT$$", 
    "previous_content": "$$STEP_3_OUTPUT$$"}}}}
    ]"""

    plan_json = ""
    try:
        plan_json = call_llm_robust(system_prompt, goal, json_mode=True)
        plan = json.loads(plan_json)
        # Validate the output structure
        if not isinstance(plan, list):
            # Handle cases where the LLM wraps the list in a dictionary (e.g., {"plan": [...]})
            if isinstance(plan, dict) and "plan" in plan and isinstance(plan["plan"], list):
                plan = plan["plan"]
            else:
                raise ValueError("Planner did not return a valid JSON list structure.")
            print("[Engine: Planner] Plan generated successfully.")
            return plan
    except Exception as e:
        print(f"[Engine: Planner] Failed to generate a valid plan. Error: {e}.RawLLM Output: {plan_json}")
        raise e
