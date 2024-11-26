# RAG-based Chatbot with COT

This project is a chatbot developed using a Retrieval-Augmented Generation (RAG) model with a Chain of Thought (CoT) methodology to assist users in querying information on WordPress sites. The chatbot retrieves relevant posts based on the user’s input, applies summarization, and generates responses.
## Project Overview

This chatbot is designed to:
- Retrieve relevant content from WordPress using the WordPress REST API.
- Generate responses by retrieving and summarizing relevant excerpts.
- Apply a Chain of Thought (CoT) methodology to enhance response coherence.
- Provide a streamlined, customizable frontend using Streamlit.

## Table of Contents

- Installation
- Key Components
- Usage
- Features
- Components
- License

## Installation

- Clone this repository:

  ```bash
      git clone https://github.com/Adya-Mishra/RAG-based-Chatbot-with-COT

- Navigate to the project directory:

  ```bash
      cd RAG-based-Chatbot-with-COT

- Install the required dependencies:
  
  ```bash
      pip install -r requirements.txt

## Key Components

- app.py:
Provides the frontend interface using Streamlit.
Displays a clean, styled input box for user queries and outputs chatbot responses.

- embeddings.py:
Manages text encoding with sentence-transformers and stores embeddings in a FAISS index.
Updates and syncs embeddings with stored content excerpts to ensure consistency.

- rag_module.py:
Contains functions for generating CoT-based responses by retrieving and summarizing relevant content.
Integrates the RAG model to combine contextually relevant WordPress data with logical response steps.

- wordpress_api.py:
Retrieves posts from a WordPress site using the REST API and ranks them by similarity to the user's query.
Uses TF-IDF and cosine similarity to filter and rank relevant content.

## Usage

- Start the application by running:
  
  ```bash
      streamlit run app.py

- Enter your query in the input box provided, and press "Get Response."
  
- The chatbot will display a response generated through a series of logical steps (CoT) for enhanced clarity and relevance.
  
## Features

- Retrieval-Augmented Generation (RAG): Enhances response relevance by retrieving content from a WordPress site and generating responses based on the most relevant excerpts.
- Chain of Thought (CoT): Breaks down the response generation into logical steps, improving coherence and alignment with user queries.
- Summarization Pipeline: Uses BART to summarize retrieved content, optimizing the response generation process.
- Customizable Frontend: A responsive and user-friendly interface created with Streamlit for easy interaction.
  
## Components

- Retrieval Process: Extracts relevant WordPress posts and summarizes them with a BART-based summarizer.
- FAISS Index: Stores and retrieves text embeddings to support efficient similarity search.
- CoT-Based Response Generation: Ensures responses are logically structured and contextually accurate.
- API Integration: Uses WordPress REST API for data retrieval and displays content with additional similarity scoring.

## License

This project is licensed under the MIT License.
