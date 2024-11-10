# test.py

import unittest
from unittest.mock import patch
from wordpress_api import retrieve_content
from embeddings import update_embeddings, rag_generate_response
from rag_module import generate_response_with_cot
from utils import clean_text, log_error
import logging


class TestWordPressAPI(unittest.TestCase):
    @patch('wordpress_api.requests.get')
    def test_retrieve_content_success(self, mock_get):
        # Mocking a successful API response
        mock_response = {
            "title": {"rendered": "Test Post Title"},
            "excerpt": {"rendered": "This is a test excerpt."},
            "link": "https://example.com/test-post"
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [mock_response]

        content = retrieve_content('AI news')
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["title"], "Test Post Title")
        self.assertEqual(content[0]["excerpt"], "This is a test excerpt.")
        self.assertEqual(content[0]["link"], "https://example.com/test-post")
    
    @patch('wordpress_api.requests.get')
    def test_retrieve_content_failure(self, mock_get):
        # Simulate an API failure (e.g., network issues)
        mock_get.return_value.status_code = 500
        content = retrieve_content('AI news')
        self.assertEqual(content, [])


class TestEmbeddings(unittest.TestCase):
    def test_update_embeddings(self):
        text = "This is a test sentence."
        # Call the function and ensure no exceptions are raised
        update_embeddings(text)  
        
        # To validate, you'd need access to FAISS index to check if embedding was added.
        # This is usually done with a mock or by checking index size.

    def test_rag_generate_response(self):
        retrieved_content = [
            {"excerpt": "This is a relevant excerpt about AI."},
            {"excerpt": "Another AI-related content snippet."}
        ]
        user_query = "Tell me about artificial intelligence."
        response = rag_generate_response(user_query, retrieved_content)
        self.assertIn("AI", response)  # Check if the response contains relevant keywords


class TestRAGModule(unittest.TestCase):
    @patch('rag_module.retrieve_content')
    def test_generate_response_with_cot(self, mock_retrieve_content):
        # Mocking content retrieval
        mock_retrieve_content.return_value = [
            {"title": "AI in Healthcare", "excerpt": "AI is transforming healthcare..."},
            {"title": "AI in Education", "excerpt": "AI is revolutionizing education..."}
        ]

        user_query = "What is AI in healthcare?"
        response = generate_response_with_cot(user_query)

        # Check if the response includes the expected content from retrieved content
        self.assertIn("AI is transforming healthcare", response)

    @patch('rag_module.retrieve_content')
    def test_generate_response_with_cot_no_content(self, mock_retrieve_content):
        # Simulate no content found
        mock_retrieve_content.return_value = []

        user_query = "What is AI in agriculture?"
        response = generate_response_with_cot(user_query)

        # Check if the response matches the "no content found" message
        self.assertEqual(response, "Sorry, I couldn't find relevant information.")


class TestUtils(unittest.TestCase):
    def test_clean_text(self):
        text = "  Hello World!  "
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "hello world!")

    def test_log_error(self):
        # Set up logging configuration to capture error logs during this test
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)
        
        with self.assertLogs(logger, level='ERROR') as log:
            log_error("Test error message")
            # Verify that "Test error message" appears in the logs
            self.assertTrue(any("Test error message" in message for message in log.output))

if __name__ == '__main__':
    unittest.main()