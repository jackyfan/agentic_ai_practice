import json
import time
from tqdm.auto import tqdm
import tiktoken
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_random_exponential
import re
import textwrap
import copy
import os
from commons.utils import initialize_clients


def create_index(pc):
    EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small
    # --- Define Index and Namespaces (assuming this is already done) ---
    INDEX_NAME = 'genai-mas-mcp-ch3'
    NAMESPACE_KNOWLEDGE = "KnowledgeStore"
    NAMESPACE_CONTEXT = "ContextLibrary"
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    # Check if index exists
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating new serverless index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric='cosine',
            spec=spec
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(1)
        print("Index created successfully. It is new and empty.")
    else:
        print(f"Index '{INDEX_NAME}' already exists. Clearing namespaces for a fresh start...")
        index = pc.Index(INDEX_NAME)
        namespaces_to_clear = [NAMESPACE_KNOWLEDGE, NAMESPACE_CONTEXT]
        for namespace in namespaces_to_clear:
            # Check if namespace exists and has vectors before deleting
            stats = index.describe_index_stats()
            if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
                print(f"Clearing namespace '{namespace}'...")
                index.delete(delete_all=True, namespace=namespace)

                # **CRITICAL FUNCTTION: Wait for deletion to complete**
                while True:
                    stats = index.describe_index_stats()
                    if namespace not in stats.namespaces or stats.namespaces[namespace].vector_count == 0:
                        print(f"Namespace '{namespace}' cleared successfully.")
                        break
                    print(f"Waiting for namespace '{namespace}' to clear...")
                    time.sleep(5)  # Poll every 5 seconds
            else:
                print(f"Namespace '{namespace}' is already empty or does not exist. Skipping.")

    # Connect to the index for subsequent operations
    index = pc.Index(INDEX_NAME)
    print(f"Connected to index '{INDEX_NAME}'.")
    return index


def data_preparation():
    # 3.Data Preparation: The Context Library (Procedural RAG)
    # -------------------------------------------------------------------------
    # We define the Semantic Blueprints derived from Chapter 1.
    # CRITICAL: We embed the 'description' (the intent), so the Librarian agent
    # can find the right blueprint based on the desired style. The 'blueprint'
    # itself is stored as metadata.

    context_blueprints = [
        {
            "id": "blueprint_suspense_narrative",
            "description": "A precise Semantic Blueprint designed to generate suspenseful and tense narratives, suitable for children's stories. Focuses on atmosphere, perceived threats, and emotional impact. Ideal for creative writing.",
            "blueprint": json.dumps({
                "scene_goal": "Increase tension and create suspense.",
                "style_guide": "Use short, sharp sentences. Focus on sensory details (sounds, shadows). Maintain a slightly eerie but age-appropriate tone.",
                "participants": [
                    {"role": "Agent", "description": "The protagonist experiencing the events."},
                    {"role": "Source_of_Threat", "description": "The underlying danger or mystery."}
                ],
                "instruction": "Rewrite the provided facts into a narrative adhering strictly to the scene_goal and style_guide."
            })
        },
        {
            "id": "blueprint_technical_explanation",
            "description": "A Semantic Blueprint designed for technical explanation or analysis. This blueprint focuses on clarity, objectivity, and structure. Ideal for breaking down complex processes, explaining mechanisms, or summarizing scientific findings.",
            "blueprint": json.dumps({
                "scene_goal": "Explain the mechanism or findings clearly and concisely.",
                "style_guide": "Maintain an objective and formal tone. Use precise terminology. Prioritize factual accuracy and clarity over narrative flair.",
                "structure": ["Definition", "Function/Operation", "Key Findings/Impact"],
                "instruction": "Organize the provided facts into the defined structure, adhering to the style_guide."
            })
        },
        {
            "id": "blueprint_casual_summary",
            "description": "A goal-oriented context for creating a casual, easy-to-read summary. Focuses on brevity and accessibility, explaining concepts simply.",
            "blueprint": json.dumps({
                "scene_goal": "Summarize information quickly and casually.",
                "style_guide": "Use informal language. Keep it brief and engaging. Imagine explaining it to a friend.",
                "instruction": "Summarize the provided facts using the casual style guide."
            })
        }
    ]
    # @title Updating the Data Loading and Processing Logic
    # -------------------------------------------------------------------------
    # Load all documents from our new directory
    knowledge_base = {}
    doc_dir = "nasa_documents"
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(doc_dir, filename), 'r') as f:
                knowledge_base[filename] = f.read()

    print(f"📚 Loaded {len(knowledge_base)} documents into the knowledge base.")  # We use sample data related to space exploration.

    knowledge_data_raw = """
    Space exploration is the use of astronomy and space technology to explore outer space. The early era of space exploration was driven by a "Space Race" between the Soviet Union and the United States. The launch of the Soviet Union's Sputnik 1 in 1957, and the first Moon landing by the American Apollo 11 mission in 1969 are key landmarks.

    The Apollo program was the United States human spaceflight program carried out by NASA which succeeded in landing the first humans on the Moon. Apollo 11 was the first mission to land on the Moon, commanded by Neil Armstrong and lunar module pilot Buzz Aldrin, with Michael Collins as command module pilot. Armstrong's first step onto the lunar surface occurred on July 20, 1969, and was broadcast on live TV worldwide. The landing required Armstrong to take manual control of the Lunar Module Eagle due to navigational challenges and low fuel.

    Juno is a NASA space probe orbiting the planet Jupiter. It was launched on August 5, 2011, and entered a polar orbit of Jupiter on July 5, 2016. Juno's mission is to measure Jupiter's composition, gravitational field, magnetic field, and polar magnetosphere to understand how the planet formed. Juno is the second spacecraft to orbit Jupiter, after the Galileo orbiter. It is uniquely powered by large solar arrays instead of RTGs (Radioisotope Thermoelectric Generators), making it the farthest solar-powered mission.

    A Mars rover is a remote-controlled motor vehicle designed to travel on the surface of Mars. NASA JPL managed several successful rovers including: Sojourner, Spirit, Opportunity, Curiosity, and Perseverance. The search for evidence of habitability and organic carbon on Mars is now a primary NASA objective. Perseverance also carried the Ingenuity helicopter.
    """

    return context_blueprints, knowledge_data_raw, knowledge_base


def create_nasa_documents():
    if not os.path.exists("nasa_documents"):
        os.makedirs("nasa_documents")

    # --- Document 1: Juno Mission ---
    juno_text = """
    The Juno mission's primary goal is to understand the origin and evolution of Jupiter. Underneath its dense cloud cover, Jupiter safeguards secrets to the fundamental processes and conditions that governed our solar system during its formation. As our primary example of a giant planet, Jupiter can also provide critical knowledge for understanding the planetary systems being discovered around other stars. Juno's specific scientific objectives include:
    1. Origin: Determine the abundance of water and constrain the planet's core mass to decide which theory of the planet's formation is correct.
    2. Atmosphere: Understand the composition, temperature, cloud motions and other properties of Jupiter's atmosphere.
    3. Magnetosphere: Map Jupiter's magnetic and gravity fields, revealing the planet's deep structure and exploring the polar magnetosphere.
    Juno is the first space mission to orbit an outer-planet from pole to pole, and the first to fly below the planet's hazardous radiation belts.
    """
    with open("nasa_documents/juno_mission_overview.txt", "w") as f:
        f.write(juno_text)

    # --- Document 2: Perseverance Rover ---
    perseverance_text = """
    The Perseverance rover's primary mission on Mars is to seek signs of ancient life and collect samples of rock and regolith (broken rock and soil) for possible return to Earth. The rover has a drill to collect core samples of the most promising rocks and soils, and sets them aside in a "cache" on the surface of Mars. The mission also provides opportunities to gather knowledge and demonstrate technologies that address the challenges of future human expeditions to Mars. These include testing a method for producing oxygen from the Martian atmosphere, identifying other resources (such as subsurface water), improving landing techniques, and characterizing weather, dust, and other potential environmental conditions that could affect future astronauts living and working on Mars. Perseverance carries the Ingenuity Helicopter, a technology demonstration to test the first powered flight on Mars.
    """
    with open("nasa_documents/perseverance_rover_tools.txt", "w") as f:
        f.write(perseverance_text)

    print("✅ Created 2 sample NASA document files in the 'nasa_documents' directory.")


def chunk_text(text, chunk_size=400, overlap=50):
    """Chunks text based on token count with overlap (Best practice for RAG)."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        # Basic cleanup
        chunk_text = chunk_text.replace("\n", " ").strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings_batch(texts, client, model):
    """Generates embeddings for a batch of texts using OpenAI, with retries."""
    # OpenAI expects the input texts to have newlines replaced by spaces
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def upsert_index(index, context_blueprints, knowledge_data_raw, knowledge_base, client, embedding_model):
    # @title 6.Process and Upload Data
    # -------------------------------------------------------------------------
    NAMESPACE_CONTEXT = "ContextLibrary"
    NAMESPACE_KNOWLEDGE = "KnowledgeStore"
    # --- 6.1. Context Library ---
    print(f"\nProcessing and uploading Context Library to namespace: {NAMESPACE_CONTEXT}")

    vectors_context = []
    for item in tqdm(context_blueprints):
        # We embed the DESCRIPTION (the intent)
        embedding = get_embeddings_batch([item['description']], client, embedding_model)[0]
        vectors_context.append({
            "id": item['id'],
            "values": embedding,
            "metadata": {
                "description": item['description'],
                # The blueprint itself (JSON string) is stored as metadata
                "blueprint_json": item['blueprint']
            }
        })

    # Upsert data
    if vectors_context:
        index.upsert(vectors=vectors_context, namespace=NAMESPACE_CONTEXT)
        print(f"Successfully uploaded {len(vectors_context)} context vectors.")

    # --- 6.2. Knowledge Base ---
    print(f"\nProcessing and uploading Knowledge Base to namespace: {NAMESPACE_KNOWLEDGE}")
    batch_size = 100
    total_vectors_uploaded = 0

    # Chunk the knowledge data
    knowledge_chunks = chunk_text(knowledge_data_raw)
    print(f"Created {len(knowledge_chunks)} knowledge chunks.")

    vectors_knowledge = []
    batch_size = 100  # Process in batches

    for doc_name, doc_content in knowledge_base.items():
        print(f"  - Processing document: {doc_name}")
        # Chunk the document content
        knowledge_chunks = chunk_text(doc_content)

        # Process in batches
        for i in tqdm(range(0, len(knowledge_chunks), batch_size), desc=f"  Uploading {doc_name}"):
            batch_texts = knowledge_chunks[i:i + batch_size]
            batch_embeddings = get_embeddings_batch(batch_texts,client, embedding_model)

            batch_vectors = []
            for j, embedding in enumerate(batch_embeddings):
                chunk_id = f"{doc_name}_chunk_{total_vectors_uploaded + j}"

                # CRITICAL UPGRADE: Add the 'source' document name to the metadata
                batch_vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": batch_texts[j],
                        "source": doc_name  # This is the key to verifiability
                    }
                })

            # Upsert the batch
            index.upsert(vectors=batch_vectors, namespace=NAMESPACE_KNOWLEDGE)
        total_vectors_uploaded += len(knowledge_chunks)

    print(f"Successfully uploaded {len(knowledge_chunks)} knowledge vectors.")


def pipeline():
    EMBEDDING_MODEL = "text-embedding-v2"
    client, pc = initialize_clients()
    # 创建NASA 文档
    create_nasa_documents()
    index = create_index(pc)
    context_blueprints, knowledge_data_raw, knowledge_base = data_preparation()
    upsert_index(index, context_blueprints, knowledge_data_raw, knowledge_base, client, EMBEDDING_MODEL)


if __name__ == "__main__":
    pipeline()
