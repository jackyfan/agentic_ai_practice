from .helpers import (
    create_mcp_message,
    call_llm_robust, query_pinecone, helper_sanitize_input)
from .utils import initialize_clients
import json
import logging


def researcher_agent(mcp_message, client, index, generation_model, embedding_model, namespace_knowledge):
    """
   Retrieves and synthesizes factual information from the Knowledge Base.
   """
    logging.info("\n[Researcher] Activated. Investigating topic...")
    try:
        topic = mcp_message['content']['topic_query']
        if not topic:
            raise ValueError("Researcher requires 'topic_query' in the input content.")
        # Query Pinecone Knowledge Namespace
        results = query_pinecone(query_text=topic, namespace=namespace_knowledge,
                                 top_k=3, index=index, client=client, embedding_model=embedding_model)
        # Sanitize and Prepare Source Texts
        sanitized_texts = []
        sources = set()
        for match in results:
            try:
                clean_text = helper_sanitize_input(
                    match['metadata']['text'])
                sanitized_texts.append(clean_text)
                if 'source' in match['metadata']:
                    sources.add(match['metadata']['source'])
            except ValueError as e:
                logging.warning(f"[Researcher] A retrieved chunk failed sanitization and was skipped.Reason: {e}")
                continue

        if not sanitized_texts:
            logging.error("[Researcher] All retrieved chunks failed sanitization.Aborting.")
            return create_mcp_message("Researcher", {"answer": "Could not generate a reliable answer as retrieved data was suspect.", "sources": []})

        # Synthesize the findings (Retrieve-and-Synthesize)
        logging.info(f"[Researcher] Found {len(sanitized_texts)} relevant chunks. Synthesizing answer with citations...")
        source_texts = [match['metadata']['text'] for match in results]
        system_prompt = """You are an expert research synthesis AI. Your task is 
                        to provide a clear, factual answer to the user's topic based *only* on the 
                        provided source texts. After the answer, you MUST provide a "Sources" section 
                        listing the unique source document names you used."""

        source_material = "\n\n---\n\n".join(sanitized_texts)
        user_prompt = f"Topic: {topic}\n\nSources:\n{source_material}\n\n--- \nSynthesize your answer and list the source documents now."
        findings = call_llm_robust(system_prompt, user_prompt, client=client, generation_model=generation_model)
        # We can also append the sources we found programmatically for robustness
        final_output = f"{findings}\n\n**Sources:**\n" + "\n".join(
            [f"- {s}" for s in sorted(list(sources))])
        return create_mcp_message(
            "Researcher", {"answer_with_sources": final_output}
        )
    except Exception as e:
        logging.error(f"[Researcher] An error occurred: {e}")
        raise e


def writer_agent(mcp_message, client, generation_model):
    """
    Combines the factual research with the semantic blueprint to generate the final output.
    """
    logging.info("\n[创作智能体] Activated. Applying blueprint to facts...")
    try:
        blueprint_data = mcp_message['content'].get('blueprint')
        facts_data = mcp_message['content'].get('facts')
        previous_content = mcp_message['content'].get('previous_content')

        # Extract the actual strings, handling both dict and raw string inputs
        blueprint_json_string = blueprint_data.get('blueprint') \
            if isinstance(blueprint_data, dict) else blueprint_data
        # ROBUST LOGIC (for Chapter 6) for handling 'facts' or 'summary'
        facts = None
        if isinstance(facts_data, dict):
            # First, try to get 'facts' (from Researcher)
            facts = facts_data.get('facts')
            # If that fails, try to get 'summary' (from Summarizer)
            if facts is None:
                facts = facts_data.get('summary')
            if facts is None:
                facts = facts_data.get('answer_with_sources')
        elif isinstance(facts_data, str):
            facts = facts_data

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

        system_prompt = f"""You are an expert content generation AI.
               Your task is to generate content based on the provided SOURCE MATERIAL.
               Crucially, you MUST structure, style, and constrain your output according to the rules defined in the SEMANTIC BLUEPRINT provided below.

               --- SEMANTIC BLUEPRINT (JSON) ---
               {blueprint_json_string}
               --- END SEMANTIC BLUEPRINT ---

               Adhere strictly to the blueprint's instructions, style guides, and goals. The blueprint defines HOW you write; the source material defines WHAT you write about.
               """

        user_prompt = f"""
               --- SOURCE MATERIAL ({source_label}) ---
               {source_material}
               --- END SOURCE MATERIAL ---

               Generate the content now, following the blueprint precisely.
               """
        # UPGRADE: Pass all dependencies to the robust LLM call.
        final_output = call_llm_robust(
            system_prompt,
            user_prompt,
            client=client,
            generation_model=generation_model
        )
        return create_mcp_message("Writer", final_output)
    except Exception as e:
        logging.error(f"[创作智能体] An error occurred: {e}")
        raise e


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


def context_librarian_agent(mcp_message, client, index, embedding_model, namespace_context):
    """
     Retrieves the appropriate Semantic Blueprint from the Context Library.
     """
    print("\\n[上下文管理员] 已激活. Analyzing intent...")
    try:
        requested_intent = mcp_message['content']['intent_query']
        if not requested_intent:
            raise ValueError("Librarian requires 'intent_query' in the input content.")
        results = query_pinecone(
            query_text=requested_intent,
            namespace=namespace_context,
            top_k=1,
            index=index,
            client=client,
            embedding_model=embedding_model
        )
        if results:
            match = results[0]
            print(f"[上下文管理员] Found blueprint '{match['id']}' (Score: {match['score']: .2f})")
            blueprint_json = match['metadata']['blueprint_json']
            content = {"blueprint": blueprint_json}
        else:
            print("[上下文管理员] No specific blueprint found. Returning default.")
            content = {"blueprint": json.dumps({"instruction": "Generate the content neutrally."})}
        return create_mcp_message("Librarian", content)
    except Exception as e:
        logging.error(f"[Librarian] An error occurred: {e}")
        raise e


def summarizer_agent(mcp_message, client, generation_model):
    """
     Reduces a large text to a concise summary based on an objective.
     Acts as a gatekeeper to manage token counts and costs.
     """
    logging.info("[摘要器智能体] Activated. Reducing context...")
    try:
        # Unpack the inputs from the MCP message
        text_to_summarize = mcp_message['content'].get('text_to_summarize')
        summary_objective = mcp_message['content'].get('summary_objective')
        # The agent validates that it has received the necessary inputs before proceeding.
        if not text_to_summarize or not summary_objective:
            raise ValueError("Summarizer requires 'text_to_summarize' and 'summary_objective' in the input content.")
        # Define the prompts for the LLM
        system_prompt = """You are an expert summarization AI. 
            Your task is to reduce the provided text to its essential points, guided by the user's specific objective. 
            The summary must be concise, accurate, and directly address the stated goal."""

        user_prompt = f"""--- OBJECTIVE ---\n{summary_objective}\n\n
        --- TEXT TO SUMMARIZE ---\n{text_to_summarize}\n--- END TEXT 
        ---\n\nGenerate the summary now."""
        # The agent calls the robust LLM helper function and returns the result.
        # Call the hardened LLM helper to perform the summarization
        summary = call_llm_robust(
            system_prompt,
            user_prompt,
            client=client,
            generation_model=generation_model
        )
        # Return the summary in the standard MCP format
        return create_mcp_message("Summarizer", {"summary": summary})
    except Exception as e:
        logging.error(f"[Summarizer] An error occurred: {e}")
        raise e


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
