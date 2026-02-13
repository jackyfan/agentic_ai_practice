from helpers import (
    create_mcp_message,
    call_llm_robust,
    call_llm, query_pinecone, display_mcp)
from utils import initialize_clients
import json


def researcher_agent(mcp_message, client):
    """
   Retrieves and synthesizes factual information from the Knowledge Base.
   """
    print("\n[Researcher] Activated. Investigating topic...")
    topic = mcp_message['content']['topic_query']
    if not topic:
        raise ValueError("Researcher requires 'topic_query' in the input content.")

    NAMESPACE_KNOWLEDGE = "KnowledgeStore"
    # Query Pinecone Knowledge Namespace
    results = query_pinecone(topic, NAMESPACE_KNOWLEDGE, top_k=3)

    if not results:
        print("[Researcher] No relevant information found.")
        return create_mcp_message("Researcher", {"facts": "No data found."})

    # Synthesize the findings (Retrieve-and-Synthesize)
    print(f"[Researcher] Found {len(results)} relevant chunks. Synthesizing...")
    source_texts = [match['metadata']['text'] for match in results]

    system_prompt = """You are an expert research synthesis AI.
        Synthesize the provided source texts into a concise, bullet-pointed summary relevant to the user's topic. 
        Focus strictly on the facts provided in the sources. Do not add outside information."""

    user_prompt = f"Topic: {topic}\n\nSources:\n" + "\n\n---\n\n".join(source_texts)

    findings = call_llm_robust(system_prompt, user_prompt, client)

    return create_mcp_message("Researcher", findings)


def writer_agent(mcp_message, client):
    """
    Combines the factual research with the semantic blueprint to generate the final output.
    """
    print("\n[Writer] Activated. Applying blueprint to facts...")

    facts = mcp_message['content']['facts']
    # The blueprint is passed as a JSON string
    blueprint_json_string = mcp_message['content']['blueprint']
    previous_content = mcp_message['content'].get('previous_content')

    if not blueprint_json_string:
        raise ValueError("Writer requires 'blueprint' in the input content.")

    if facts:
        source_material = facts
        source_label = "RESEARCH FINDINGS"
    elif previous_content:
        source_material = previous_content
        source_label = "PREVIOUS CONTENT (For Rewriting)"
    else:
        raise ValueError("Writer requires either 'facts' or 'previous_content'.")

    # The Writer's System Prompt incorporates the dynamically retrieved blueprint
    system_prompt = f"""You are an expert content generation AI.
    Your task is to generate content based on the provided RESEARCH FINDINGS.
    Crucially, you MUST structure, style, and constrain your output according to the rules defined in the SEMANTIC BLUEPRINT provided below.

    --- SEMANTIC BLUEPRINT (JSON) ---
    {blueprint_json_string}
    --- END SEMANTIC BLUEPRINT ---

    Adhere strictly to the blueprint's instructions, style guides, and goals. The blueprint defines HOW you write; the research defines WHAT you write about.
    """

    user_prompt = f"""
    --- SOURCE MATERIAL ({source_label}) ---
    {source_material}
     --- END SOURCE MATERIAL ---
     Generate the content now, following the blueprint precisely.
     """

    # Generate the final content (slightly higher temperature for potential creativity)
    final_output = call_llm(system_prompt, user_prompt, client)

    return create_mcp_message("Writer", final_output)


# --- Agent 3: The Validator ---
def validator_agent(mcp_input, client):
    """This agent fact-checks a draft against a source summary."""
    print("\n[验证Agent已激活]")
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
    validation_result = call_llm_robust(system_prompt, validation_context, client)
    print(f"验证已完成，结果: {validation_result}")
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


def agent_context_librarian(mcp_message, client):
    """
     Retrieves the appropriate Semantic Blueprint from the Context Library.
     """
    print("\\n[上下文管理员] 已激活. Analyzing intent...")
    requested_intent = mcp_message['content']['intent_query']
    if not requested_intent:
        raise ValueError("Librarian requires 'intent_query' in the input content.")
    NAMESPACE_CONTEXT = "ContextLibrary"
    results = query_pinecone(requested_intent, NAMESPACE_CONTEXT, top_k=1)
    if results:
        match = results[0]
        print(f"[上下文管理员] Found blueprint '{match['id']}' (Score: {match['score']: .2f})")
        blueprint_json = match['metadata']['blueprint_json']
        content = {"blueprint": blueprint_json}
    else:
        print("[上下文管理员] No specific blueprint found. Returning default.")
        content = {"blueprint": json.dumps({"instruction": "Generate the contentneutrally."})}
    return create_mcp_message("Librarian", content)


def orchestrator(high_level_goal):
    """
    Manages the workflow of the Context-Aware MAS.
    """
    print("=" * 50)
    print(f"[编排器] Goal Received: '{high_level_goal}'")
    print("=" * 50)
    client, pc = initialize_clients()
    # --- Step 1: Orchestrator plans and calls the Researcher Agent ---
    print("\n[Orchestrator] Analyzing Goal...")
    analysis_system_prompt = """You are an expert goal analyst. Analyze the 
    user's high-level goal and extract two components:
     1. 'intent_query': A descriptive phrase summarizing the desired style, tone, 
    or format, optimized for searching a context library (e.g., "suspenseful narrative 
    blueprint", "objective technical explanation structure").
     2. 'topic_query': A concise phrase summarizing the factual subject matter 
    required (e.g., "Juno mission objectives and power", "Apollo 11 landing details").
     Respond ONLY with a JSON object containing these two keys."""
    # We request JSON mode for reliable parsing
    analysis_result = call_llm(
        system_prompt=analysis_system_prompt,
        user_prompt=high_level_goal,
        client=client,
        json_mode=True)

    try:
        analysis = json.loads(analysis_result)
        intent_query = analysis['intent_query']
        topic_query = analysis['topic_query']
    except (json.JSONDecodeError, KeyError):
        print(f"[Orchestrator] Error: Could not parse analysis JSON. Raw Analysis: {analysis_result}. Aborting.")
        return

    print(f"Orchestrator: Intent Query: '{intent_query}'")
    print(f"Orchestrator: Topic Query: '{topic_query}'")

    # Step 1: Get the Context Blueprint (Procedural RAG)
    mcp_to_librarian = create_mcp_message(
        sender="Orchestrator",
        content={"intent_query": intent_query}
    )
    # display_mcp(mcp_to_librarian, "Orchestrator -> Librarian")
    mcp_from_librarian = agent_context_librarian(mcp_to_librarian)
    display_mcp(mcp_from_librarian, "Librarian -> Orchestrator")
    context_blueprint = mcp_from_librarian['content'].get('blueprint')
    if not context_blueprint: return
    # Step 2: Get the Factual Knowledge (Factual RAG)
    mcp_to_researcher = create_mcp_message(
        sender="Orchestrator",
        content={"topic_query": topic_query}
    )
    # display_mcp(mcp_to_researcher, "Orchestrator -> Researcher")
    mcp_from_researcher = researcher_agent(mcp_to_researcher, client)
    display_mcp(mcp_from_researcher, "Researcher -> Orchestrator")

    research_findings = mcp_from_researcher['content'].get('facts')
    if not research_findings: return

    # Step 3: Generate the Final Output
    # Combine the outputs for the Writer Agent
    writer_task = {
        "blueprint": context_blueprint,
        "facts": research_findings
    }

    mcp_to_writer = create_mcp_message(
        sender="Orchestrator",
        content=writer_task
    )

    # display_mcp(mcp_to_writer, "Orchestrator -> Writer")
    mcp_from_writer = writer_agent(mcp_to_writer, client)
    display_mcp(mcp_from_writer, "Writer -> Orchestrator")
    final_result = mcp_from_writer['content'].get('output')
    print("\n=== [Orchestrator] Task Complete ===")
    return final_result


def final_orchestrator(initial_goal):
    """
    Manages the multi-agent workflow to achieve a high-level goal.
    """
    print("=" * 50)
    print(f"[编排器] Goal Received: '{initial_goal}'")
    print("=" * 50)
    client, _ = initialize_clients()
    # --- Step 1: Orchestrator plans and calls the Researcher Agent ---
    print("\n[编排器]任务1: Research. Delegating to Researcher Agent.")
    research_topic = "Mediterranean Diet"

    mcp_to_researcher = create_mcp_message(
        sender="Orchestrator",
        content=research_topic
    )

    mcp_from_researcher = researcher_agent(mcp_to_researcher, client)

    if not validate_mcp_message(mcp_from_researcher) or not mcp_from_researcher['content']:
        print("Workflow failed due to invalid or empty message from Researcher.")
        return
    research_summary = mcp_from_researcher['content']
    print("\n[编排器] Research complete. Received summary:")
    print("-" * 20)
    print(research_summary)
    print("-" * 20)

    # --- Step 2 & 3: Iterative Writing and Validation Loop ---
    final_output = "Could not produce a validated article."
    max_revisions = 2
    for i in range(max_revisions):
        print(f"\n[编排器] Writing Attempt {i + 1}/{max_revisions}")

        writer_context = research_summary
        if i > 0:
            writer_context += f"\n\nPlease revise the previous draft based on this feedback: {validation_result}"

        mcp_to_writer = create_mcp_message(sender="Orchestrator", content=writer_context)
        mcp_from_writer = writer_agent(mcp_to_writer, client)

        if not validate_mcp_message(mcp_from_writer) or not mcp_from_writer['content']:
            print("Aborting revision loop due to invalid message from Writer.")
            break
        draft_post = mcp_from_writer['content']

        # --- Validation Step ---
        print("\n[编排器] Draft received. Delegating to Validator Agent.")
        validation_content = {"summary": research_summary, "draft": draft_post}
        mcp_to_validator = create_mcp_message(sender="Orchestrator", content=validation_content)
        mcp_from_validator = validator_agent(mcp_to_validator, client)

        if not validate_mcp_message(mcp_from_validator) or not mcp_from_validator['content']:
            print("Aborting revision loop due to invalid message from Validator.")
            break
        validation_result = mcp_from_validator['content']

        if "pass" in validation_result.lower():
            print("\n[编排器] Validation PASSED. Finalizing content.")
            final_output = draft_post
            break
        else:
            print(f"\n[编排器] Validation FAILED. Feedback: {validation_result}")
            if i < max_revisions - 1:
                print("Requesting revision.")
            else:
                print("Max revisions reached. Workflow failed.")

    # --- Step 4: Final Presentation ---
    print("\n" + "=" * 50)
    print("[编排器] Workflow Complete. Final Output:")
    print("=" * 50)
    print(final_output)


if __name__ == "__main__":
    print("********  1: SUSPENSEFUL NARRATIVE **********")
    goal_1 = "Write a short, suspenseful scene for a children's story about the Apollo 11 moon landing, highlighting the danger."
    result_1 = orchestrator(goal_1)

    print("\n******** FINAL OUTPUT 1 **********\n")
    print(result_1)

    print("\n\n" + "=" * 50 + "\n\n")
