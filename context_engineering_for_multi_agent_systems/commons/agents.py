from helpers import call_llm, create_mcp_message


def researcher_agent(mcp_input):
    """
    This agent takes a research topic, finds information, and returns a summary.
    """
    print("\n[Researcher Agent Activated]")
    simulated_database = {
        "mediterranean diet": """The Mediterranean diet is rich in fruits,
           vegetables, whole grains, olive oil, and fish.Studies show it is associated with
           a lower risk of heart disease, improved brain health, and a longer lifespan.Key
           components include monounsaturated fats and antioxidants."""
    }
    research_topic = mcp_input['content']
    research_result = simulated_database.get(research_topic.lower(),
                                             "No information found on this topic.")
    system_prompt = """You are a research analyst. Your task is to synthesize the 
       provided information into 3-4 concise bullet points. Focus on the key findings."""
    summary = call_llm(system_prompt, research_result)
    print(f"Research summary created for: '{research_topic}'")
    return create_mcp_message(
        sender="ResearcherAgent",
        content=summary,
        metadata={"source": "Simulated Internal DB"}
    )


def writer_agent(mcp_input):
    """
    This agent takes research findings and writes a short blog post.
     """
    print("\n[Writer Agent Activated]")
    research_summary = mcp_input['content']
    system_prompt = """You are a skilled content writer for a health and wellness blog. 
    Your tone is engaging, informative, and encouraging. 
    Your task is to take the following research points and write a short, 
    appealing blog post (approx. 150 words) with a catchy title."""
    blog_post = call_llm(system_prompt, research_summary)
    print("Blog post drafted.")
    return create_mcp_message(
        sender="WriterAgent",
        content=blog_post,
        metadata={"word_count": len(blog_post.split())}
    )

def orchestrator(initial_goal):
    """
    Manages the multi-agent workflow to achieve a high-level goal.
    """
    print("=" * 50)
    print(f"[Orchestrator] Goal Received: '{initial_goal}'")
    print("=" * 50)

    # --- Step 1: Orchestrator plans and calls the Researcher Agent ---
    print("\n[Orchestrator] Task 1: Research. Delegating to Researcher Agent.")
    research_topic = "Mediterranean Diet"

    mcp_to_researcher = create_mcp_message(
        sender="Orchestrator",
        content=research_topic
    )

    mcp_from_researcher = researcher_agent(mcp_to_researcher)
    print("\n[Orchestrator] Research complete. Received summary:")
    print("-" * 20)
    print(mcp_from_researcher['content'])
    print("-" * 20)

    # --- Step 2: Orchestrator calls the Writer Agent ---
    print("\n[Orchestrator] Task 2: Write Content. Delegating to Writer Agent.")
    mcp_to_writer = create_mcp_message(
        sender="Orchestrator",
        content=mcp_from_researcher['content']
    )

    mcp_from_writer = writer_agent(mcp_to_writer)
    print("\n[Orchestrator] Writing complete.")

    # --- Step 3: Orchestrator presents the final result ---
    final_output = mcp_from_writer['content']
    print("\n" + "=" * 50)
    print("[Orchestrator] Workflow Complete. Final Output:")
    print("=" * 50)
    print(final_output)





if __name__ == "__main__":
    # @title 5. Run the System
    # ------------------------------------------------------------------------
    # Let's give our Orchestrator a high-level goal and watch the agent team work.
    # ------------------------------------------------------------------------
    user_goal = "Create a blog post about the benefits of the Mediterranean diet."
    orchestrator(user_goal)
