from helpers import call_llm,create_mcp_message
from utils import initialize_clients


def researcher_agent(mcp_prompt):
    """
    This agent takes a research topic, finds information, and returns a summary.
    """
    print("\n[Researcher Agent Activated]")


def writer_agent(mcp_prompt):
    """
    This agent takes research findings and writes a short blog post.
     """
    print("\n[Writer Agent Activated]")


def main():
    simulated_database = {
        "mediterranean diet": """The Mediterranean diet is rich in fruits,
        vegetables, whole grains, olive oil, and fish.Studies show it is associated with
        a lower risk of heart disease, improved brain health, and a longer lifespan.Key
        components include monounsaturated fats and antioxidants."""
    }
    mcp_input = create_mcp_message(
        sender="Orchestrator",
        content="Mediterranean diet",
        metadata={"task_id": "T-123", "priority": "high"}
    )
    research_topic = mcp_input['content']
    research_result = simulated_database.get(research_topic.lower(),
                                             "No information found on this topic.")
    print(research_result)
    system_prompt = """You are a research analyst. Your task is to synthesize the 
    provided information into 3-4 concise bullet points. Focus on the key findings."""
    client = initialize_clients()
    summary = call_llm(system_prompt, research_result,client)
    print(f"Research summary created for: '{research_topic}'")
    print(f"Research summary results is : '{summary}'")


if __name__ == "__main__":
    main()
