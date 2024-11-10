#rag_module.py

from wordpress_api import retrieve_content, compute_similarity_score
from embeddings import rag_generate_response
from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoderTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
import torch

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the tokenizer and model for DPR
dpr_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
dpr_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

# Initialize the RagTokenizer and RagSequenceForGeneration for RAG
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Initialize the SentenceTransformer model for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)  # You can choose any suitable model from SentenceTransformers

# Initialize a summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Global variable or a session object to keep track of conversation context
conversation_context = []

def generate_response_with_cot(user_query, context=None):
    """
    Generates a response using the Chain of Thought (CoT) strategy by:
    - Retrieving content based on the user's query
    - Summarizing relevant content
    - Generating an initial response
    - Refining the response based on context and similarity to the user query

    Parameters:
    user_query (str): The query from the user.
    context (str, optional): Previous context for refining the response.

    Returns:
    str: The final response, with thought steps interwoven throughout the main response.
    """
    # Step 1: Retrieve content relevant to the user query
    retrieved_content = retrieve_content(user_query)
    if not retrieved_content:
        return "Sorry, I couldn't find relevant information."

    # Step 2: Summarize each passage for conciseness
    summarized_content = []
    for content in retrieved_content:
        try:
            summary = summarizer(content["excerpt"], max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
            summarized_content.append({"title": content["title"], "summary": summary, "link": content["link"]})
        except Exception as e:
            print(f"Error in summarizing content: {e}")
            continue

    # Step 3: Filter summarized content by similarity to the user query
    filtered_content = filter_by_similarity(summarized_content, user_query)

    # Initialize the response with an introduction
    response = ["Here's a detailed response based on your query:"]

    # Step 4: Generate the initial response using the RAG model and add as the first thought step
    initial_response = rag_generate_response(user_query, filtered_content)
    response.append(f"\n\nInitial response: {initial_response}")

    # Step 5: Interleave thought steps for a more coherent answer
    response.append("\n\nLet’s break down the query into manageable components to address all relevant aspects.")
    for item in filtered_content:
        response.append(f"\nTopic: {item['title']}\nSummary: {item['summary']}")

    response.append("\nNow, let’s consider the broader implications and refine details further for clarity.")

    # Final thought step conclusion
    response.append("\nIn conclusion, the response balances both the query and relevant context to ensure the answer is comprehensive and informed.")

    # Combine all parts into a coherent response
    final_response = " ".join(response) + "\n\nLet me know if you need further clarification."

    # Update the conversation context for continuity
    update_conversation_context(user_query, final_response)

    return final_response


def filter_by_similarity(summarized_content, user_query, threshold=0.7):
    """
    Filters summarized content based on similarity to the user query.

    Parameters:
    summarized_content (list): A list of summarized content dictionaries.
    user_query (str): The query from the user.
    threshold (float, optional): The similarity threshold for filtering relevant content (default is 0.7).

    Returns:
    list: A list of relevant content that exceeds the similarity threshold.
    """
    relevant_content = []
    for item in summarized_content:
        score = compute_similarity_score(item["summary"], user_query)
        if score > threshold:
            relevant_content.append(item)
    return relevant_content if relevant_content else summarized_content  # Return all if none are above threshold

def refine_thought_steps(initial_response, context):
    """
    Refines the initial response by incorporating the provided context.
    Implements a Chain of Thought (CoT) strategy to break down the reasoning into logical steps.

    Parameters:
    initial_response (str): The initial response generated.
    context (str, optional): The previous conversation context to refine the response.

    Returns:
    list: A list of refined thought steps to create a detailed response.
    """
    thought_steps = [
        f"\n\n\nInitial response: {initial_response}",
        "Let’s dissect the query into manageable components to ensure we address all relevant factors step by step.",
        "We will now consider the broader implications and refine the details further for greater clarity.",
        "In conclusion, the key point here is that we are addressing both the query and context in a balanced manner, providing the most informed answer possible."
    ]

    # Optional: Include context in refinement if available
    if context:
        thought_steps.insert(1, f"Context integration: Based on previous discussions like '{context}', refining the response to be more relevant.")
    
    return thought_steps

def filter_relevant_thought_steps(thought_steps, user_query, threshold=0.6):
    """
    Filters thought steps for relevance based on query similarity.

    Parameters:
    thought_steps (list): A list of thought steps to be filtered.
    user_query (str): The user's query.
    threshold (float, optional): The similarity threshold for relevance (default is 0.6).

    Returns:
    list: A list of thought steps that are relevant to the user's query.
    """
    relevant_steps = []
    for step in thought_steps:
        score = compute_similarity_score(step, user_query)
        if score > threshold:
            relevant_steps.append(step)
    return relevant_steps if relevant_steps else thought_steps

def combine_thought_steps(thought_steps):
    """
    Combines thought steps into a coherent conversational response.

    Parameters:
    thought_steps (list): A list of refined thought steps.

    Returns:
    str: A string that combines the initial response and additional thought steps into a coherent response.
    """
    # Identify the initial response separately
    initial_response = thought_steps[0] if thought_steps else "Here's the answer based on what I've found."
    
    # Combine remaining thought steps with improved clarity
    additional_thoughts = " ".join(thought_steps[1:])
    refined_response = f"{initial_response}\n\nFurther insights:\n{additional_thoughts}"
    
    # Final response formatting
    final_response = f"{refined_response}\n\nLet me know if you need further clarification."
    return final_response


def update_conversation_context(user_query, response):
    """
    Updates the conversation context with the latest user query and response.
    Ensures the context stays relevant for future queries.

    Parameters:
    user_query (str): The latest query from the user.
    response (str): The corresponding response from the bot.
    """
    global conversation_context
    conversation_context.append({"query": user_query, "response": response})
    if len(conversation_context) > 10:  # Limit context for memory management
        conversation_context = conversation_context[-10:]

def get_conversation_context():
    """
    Retrieves the current conversation context as a formatted string.

    Returns:
    str: The formatted conversation context.
    """
    return "\n".join([f"User: {item['query']}\nBot: {item['response']}" for item in conversation_context])