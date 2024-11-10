# wordpress_api.py

import requests
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging for error tracking
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_similarity_score(user_query, texts):
    """
    Computes the similarity score between the query and a list of texts using TF-IDF and cosine similarity.
    
    Args:
    query (str): The search term or question.
    texts (list): List of text snippets to compare with the query.
    
    Returns:
    list: A list of similarity scores.
    """
    # Ensure texts is a list (if it's not already)
    if isinstance(texts, str):  # If texts is a string, make it a list
        texts = [texts]
    elif not isinstance(texts, list):  # If texts is not a string or list, raise an error
        raise TypeError("Expected 'texts' to be a string or a list of strings")
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_query] + texts)  # First item is query, others are texts
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

def retrieve_content(user_query, num_posts=5):
    """
    Retrieves content from the WordPress API based on the user's query.
    Filters and ranks retrieved content by similarity to the query.
    
    Args:
    user_query (str): The search term to find related posts.
    num_posts (int): The number of top relevant posts to return. Defaults to 5.
    
    Returns:
    list: A list of dictionaries containing post titles, excerpts, and links.
    """
    url = "https://towardsai.net/wp-json/wp/v2/posts"
    params = {'search': user_query, 'per_page': 10}  # Retrieve more posts initially to allow ranking
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print("Raw response content:", response.content)  # Debugging response
        response.raise_for_status()
        
        try:
            posts = response.json()
        except ValueError as e:
            logging.error(f"Failed to parse JSON: {e}")
            return []
        
        if not isinstance(posts, list):
            raise ValueError("Expected a list of posts, but got a different format.")
        
        # Extract title and excerpt for similarity ranking
        content = []
        texts_for_ranking = []
        
        for post in posts:
            try:
                title = post["title"]["rendered"]
                excerpt = post.get("excerpt", {}).get("rendered", "")
                combined_text = title + " " + excerpt
                texts_for_ranking.append(combined_text)
                
                content.append({
                    "title": title,
                    "excerpt": excerpt,
                    "link": post["link"]
                })
            except KeyError as e:
                logging.error(f"Missing expected key in post data: {e}")
                continue
        
        # Calculate similarity scores
        if texts_for_ranking:
            scores = compute_similarity_score(user_query, texts_for_ranking)
            for idx, post in enumerate(content):
                post["similarity_score"] = scores[idx]
            
            # Sort content by similarity scores in descending order
            content = sorted(content, key=lambda x: x["similarity_score"], reverse=True)
        
        # Return only the top num_posts most relevant results
        return content[:num_posts]
    
    except requests.exceptions.Timeout:
        logging.error("Request timed out while fetching posts.")
        return []
    except requests.exceptions.ConnectionError:
        logging.error("Network connection error while fetching posts.")
        return []
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Status code: {response.status_code}")
        return []
    except requests.exceptions.RequestException as req_err:
        logging.error(f"An error occurred while making the request: {req_err}")
        return []
    except ValueError as val_err:
        logging.error(f"Data format error: {val_err}")
        return []
    except Exception as err:
        logging.error(f"An unexpected error occurred: {err}")
        return [] 