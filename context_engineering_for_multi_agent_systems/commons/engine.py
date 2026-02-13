from helpers import call_llm_robust, create_mcp_message
import json, copy, time
from registry import AgentRegistry
from utils import initialize_clients


def planner(goal, capabilities,client):
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
        plan_json = call_llm_robust(system_prompt, goal, client, json_mode=True)
        plan = json.loads(plan_json)
        # Validate the output structure
        if not isinstance(plan, list):
            # Handle cases where the LLM wraps the list in a dictionary (e.g., {"plan": [...]})
            if isinstance(plan, dict) and "plan" in plan and isinstance(plan["plan"], list):
                plan = plan["plan"]
            else:
                raise ValueError("Planner did not return a valid JSON list structure.")
        print("[引擎:规划器] Plan generated successfully.")
        return plan
    except Exception as e:
        print(f"[引擎:规划器] Failed to generate a valid plan. Error: {e}.RawLLM Output: {plan_json}")
        raise e


def resolve_dependencies(input_params, state):
    """
    Helper function to replace
 placeholders with actual data from the execution state.
    This implements Context Chaining.
    """
    # Use copy.deepcopy to ensure the original plan structure is not modified
    resolved_input = copy.deepcopy(input_params)

    # Recursive function to handle potential nested structures
    def resolve(value):
        if isinstance(value, str) and value.startswith("$$") and value.endswith("$$"):
            ref_key = value[2:-2]
            if ref_key in state:
                # Retrieve the actual data (string) from the previous step's output
                print(f"[引擎:执行器] Resolved dependency {ref_key}.")
                return state[ref_key]
            else:
                raise ValueError(f"Dependency Error: Reference {ref_key} not found in execution state.")
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(resolved_input)


class ExecutionTrace:
    """Logs the entire execution flow for debugging and analysis."""

    def __init__(self, goal):
        self.goal = goal
        self.plan = None
        self.steps = []
        self.status = "Initialized"
        self.final_output = None
        self.start_time = time.time()
        self.duration = None

    def log_plan(self, plan):
        self.plan = plan

    def log_step(self, step_num, agent, planned_input, mcp_output, resolved_input):
        """Logs the details of a single execution step."""
        self.steps.append({
            "step": step_num,
            "agent": agent,
            "planned_input": planned_input,
            "resolved_context": resolved_input,
            "output": mcp_output['content']
        })

    def finalize(self, status, final_output=None):
        self.status = status
        self.final_output = final_output
        self.duration = time.time() - self.start_time


def context_engine(goal,client):
    """
     The main entry point for the Context Engine. Manages Planning and Execution.
     """
    print(f"\n=== [上下文引擎] Starting New Task ===\nGoal: {goal}\n")
    trace = ExecutionTrace(goal)
    registry = AgentRegistry()
    # Phase 1: Plan
    try:
        capabilities = registry.get_capabilities_description()
        plan = planner(goal, capabilities,client)
        trace.log_plan(plan)
    except Exception as e:
        print(f"[引擎:规划器] Planning Failed: {e}")
        trace.finalize("Failed during Planning")
        # Return the trace even in failure for debugging
        return None, trace

    # Phase 2: Execute
    # State stores the raw outputs (strings) of each step: { "STEP_X_OUTPUT": data_string }
    state = {}
    for step in plan:
        step_num = step.get("step")
        agent_name = step.get("agent")
        planned_input = step.get("input")
        print(f"\n[引擎:执行器] Starting Step {step_num}: {agent_name}")
        try:
            agent = registry.get_agent(agent_name)
            # Context Assembly: Resolve dependencies
            resolved_input = resolve_dependencies(planned_input, state)
            # Execute Agent via MCP
            # Create an MCP message with the RESOLVED input for the agent
            mcp_resolved_input = create_mcp_message(
                "Engine", resolved_input)
            mcp_output = agent(mcp_resolved_input,client)
            # Update State and Log Trace
            output_data = mcp_output["content"]
            # Store the output data (the context itself)
            state[f"STEP_{step_num}_OUTPUT"] = output_data
            trace.log_step(step_num, agent_name, planned_input,
                           mcp_output, resolved_input)
            print(f"[引擎:执行器] Step {step_num} completed.")
        except Exception as e:
            error_message = f"Execution failed at step {step_num}({agent_name}):{e}"
            print(f"[Engine: Executor] ERROR: {error_message}")
            trace.finalize(f"Failed at Step {step_num}")
            # Return the trace for debugging the failure
            return None, trace

    final_output = state.get(f"STEP_{len(plan)}_OUTPUT")
    trace.finalize("Success", final_output)
    print("\n=== [上下文引擎]任务完成 ===")
    return final_output, trace


if __name__ == "__main__":
    print("******** Example 1: STANDARD WORKFLOW (Suspenseful Narrative)** ** ** ** ** \n")
    goal_1 = """Write a short, suspenseful scene for a children's story about the Apollo 11 moon landing, 
    highlighting the danger."""
    # Run the Context Engine
    # Ensure the Pinecone index is populated (from Ch3 notebook) for this to work.
    client, _ = initialize_clients()
    result_1, trace_1 = context_engine(goal_1,client)
    if result_1:
        print("\n******** FINAL OUTPUT 1 **********\n")
        print(result_1)
        print("\n\n" + "=" * 50 + "\n\n")
    # Optional: Display the trace to see the engine's process
    print(trace_1.final_output)
