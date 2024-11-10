#embeddings.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Set up logging to output potential issues
logging.basicConfig(level=logging.INFO)

# Load the model for encoding text into embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# FAISS index to store the embeddings and perform similarity search
index = faiss.IndexFlatL2(768)  # 768-dimensional embeddings (from MPNet model)
stored_excerpts = []  # List to keep track of excerpts added to the FAISS index

def update_embeddings(text):
    """
    Update the FAISS index with new text by encoding it into embeddings.
    Args:
    - text (str): The text to encode and add to the FAISS index.
    """
    # Encode the text into a fixed-size embedding
    embedding = model.encode([text])
    embedding = np.array(embedding, dtype='float32')  # FAISS requires float32 format
    index.add(embedding)  # Add the embedding to the FAISS index
    stored_excerpts.append(text)  # Keep track of the text in the same order

def check_index_sync():
    """
    Ensure stored_excerpts remains in sync with the FAISS index to prevent retrieval inconsistencies.
    """
    if len(stored_excerpts) != index.ntotal:
        logging.warning("Index and stored_excerpts out of sync. Rebuilding index.")
        index.reset()  # Clears all embeddings in FAISS
        embeddings = np.array([model.encode([text])[0] for text in stored_excerpts], dtype='float32')
        index.add(embeddings)

def generate_cot_thought_process(user_query, relevant_excerpts):
    """
    Generate a Chain of Thought (CoT) to help guide the response generation logically.
    Args:
    - query (str): The user's query.
    - relevant_excerpts (list): The excerpts considered relevant to the query.
    
    Returns:
    str: The Chain of Thought process, leading to a coherent response.
    """
    # Start with the user's query and relevant excerpts
    cot_steps = []
    
    if relevant_excerpts:
        cot_steps.append("The system is reviewing the most relevant excerpts found:")
        cot_steps.extend(relevant_excerpts)
    
    cot_steps.append("The following conclusion is drawn based on the relevant context.")
    return "\n".join(cot_steps)

def rag_generate_response(user_query, retrieved_content):
    """
    Generate a response based on user query and retrieved content by similarity search on FAISS index.
    
    Args:
    - user_query (str): The query entered by the user.
    - retrieved_content (list of dict): Retrieved passages where each dict contains an 'excerpt' key.

    Returns:
    str: A response generated based on the most relevant excerpts.
    """
    # Ensure FAISS index and stored_excerpts are in sync
    check_index_sync()
    
    # Encode and add each new excerpt to the FAISS index
    for content in retrieved_content:
        excerpt = content.get('summary')  # Summary is used for conciseness
        if excerpt and excerpt not in stored_excerpts:
            update_embeddings(excerpt)
    
    # Encode user query for similarity search
    query_embedding = model.encode([user_query]).astype('float32')
    distances, indices = index.search(query_embedding, k=3)
    logging.info(f"indices returned: {indices[0]}")
    
    # Threshold filtering for relevant excerpts
    threshold = 0.5
    relevant_excerpts = [
        stored_excerpts[i] for i, dist in zip(indices[0], distances[0]) 
        if 0 <= i < len(stored_excerpts) and dist < threshold
    ]

    # Fallback to top excerpts if none meet the threshold
    if not relevant_excerpts:
        relevant_excerpts = [stored_excerpts[i] for i in indices[0] if 0 <= i < len(stored_excerpts)]


    
    # Generate Chain of Thought (CoT) process
    cot_process = generate_cot_thought_process(user_query, relevant_excerpts)
    
    # Join excerpts into a single response
    response = " ".join(relevant_excerpts)
    
    # Optionally, include the CoT in the response for improved explanation
    response_with_cot = f"{cot_process}\n\nFinal Response: {response}"
    return response_with_cot