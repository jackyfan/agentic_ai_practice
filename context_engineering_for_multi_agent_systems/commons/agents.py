from context_engineering_for_multi_agent_systems.commons.helpers import call_llm_robust
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

    summary = call_llm_robust(system_prompt, research_result)
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

    blog_post = call_llm_robust(system_prompt, research_summary)
    print("Blog post drafted.")
    return create_mcp_message(
        sender="WriterAgent",
        content=blog_post,
        metadata={"word_count": len(blog_post.split())}
    )


# --- Agent 3: The Validator ---
def validator_agent(mcp_input):
    """This agent fact-checks a draft against a source summary."""
    print("\n[Validator Agent Activated]")
    # Extracting the two required pieces of information
    source_summary = mcp_input['content']['summary']
    draft_post = mcp_input['content']['draft']
    system_prompt = """
     You are a meticulous fact-checker. Determine if the 'DRAFT' is factually 
    consistent with the 'SOURCE SUMMARY'.
     - If all claims in the DRAFT are supported by the SOURCE, respond with only 
    the word \"pass\".
     - If the DRAFT contains any information not in the SOURCE, respond with 
    \"fail\" and a one-sentence explanation.
     """
    validation_context = f"SOURCE SUMMARY:\n{source_summary}\n\nDRAFT:\n{draft_post}"
    validation_result = call_llm_robust(system_prompt, validation_context)
    print(f"Validation complete. Result: {validation_result}")
    return create_mcp_message(
        sender="ValidatorAgent",
        content=validation_result
    )


def validate_mcp_message(message):
    """A simple validator to check the structure of an MCP message."""
    required_keys = ["protocol_version", "sender", "content", "metadata"]
    if not isinstance(message, dict):
        print(f"MCP Validation Failed: Message is not a dictionary.")
        return False
    for key in required_keys:
        if key not in message:
            print(f"MCP Validation Failed: Missing key '{key}'")
            return False
        print(f"MCP message from {message['sender']} validated successfully.")
    return True


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

def final_orchestrator(initial_goal):
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

    if not validate_mcp_message(mcp_from_researcher) or not mcp_from_researcher['content']:
        print("Workflow failed due to invalid or empty message from Researcher.")
        return
    research_summary = mcp_from_researcher['content']
    print("\n[Orchestrator] Research complete. Received summary:")
    print("-" * 20)
    print(research_summary)
    print("-" * 20)

    # --- Step 2 & 3: Iterative Writing and Validation Loop ---
    final_output = "Could not produce a validated article."
    max_revisions = 2
    for i in range(max_revisions):
        print(f"\n[Orchestrator] Writing Attempt {i + 1}/{max_revisions}")

        writer_context = research_summary
        if i > 0:
            writer_context += f"\n\nPlease revise the previous draft based on this feedback: {validation_result}"

        mcp_to_writer = create_mcp_message(sender="Orchestrator", content=writer_context)
        mcp_from_writer = writer_agent(mcp_to_writer)

        if not validate_mcp_message(mcp_from_writer) or not mcp_from_writer['content']:
            print("Aborting revision loop due to invalid message from Writer.")
            break
        draft_post = mcp_from_writer['content']

        # --- Validation Step ---
        print("\n[Orchestrator] Draft received. Delegating to Validator Agent.")
        validation_content = {"summary": research_summary, "draft": draft_post}
        mcp_to_validator = create_mcp_message(sender="Orchestrator", content=validation_content)
        mcp_from_validator = validator_agent(mcp_to_validator)

        if not validate_mcp_message(mcp_from_validator) or not mcp_from_validator['content']:
            print("Aborting revision loop due to invalid message from Validator.")
            break
        validation_result = mcp_from_validator['content']

        if "pass" in validation_result.lower():
            print("\n[Orchestrator] Validation PASSED. Finalizing content.")
            final_output = draft_post
            break
        else:
            print(f"\n[Orchestrator] Validation FAILED. Feedback: {validation_result}")
            if i < max_revisions - 1:
                print("Requesting revision.")
            else:
                print("Max revisions reached. Workflow failed.")

    # --- Step 4: Final Presentation ---
    print("\n" + "=" * 50)
    print("[Orchestrator] Workflow Complete. Final Output:")
    print("=" * 50)
    print(final_output)

if __name__ == "__main__":
    # @title 6.Run the Final, Robust System
    user_goal = "Create a blog post about the benefits of the Mediterranean diet."
    final_orchestrator(user_goal)
