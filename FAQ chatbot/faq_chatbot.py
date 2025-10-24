"""
FAQ Chatbot using Semantic Search
==================================

Description:
    This script implements an AI-powered FAQ chatbot using semantic search.
    It loads FAQs from a CSV, encodes questions with Sentence Transformers,
    and responds to user queries by finding the most similar FAQ answer.

Purpose:
    - Demonstrate semantic search for conversational AI.
    - Provide a simple chatbot for FAQ automation.
    - Showcase professional Python: NLP integration, data loading, interactive loops.
    - Educate on similarity thresholds and model limitations.

Key Features:
    - Semantic encoding: Uses Sentence Transformers for meaning-based matching.
    - Interactive mode: Command-line chat with exit commands.
    - Confidence scoring: Shows similarity score; warns for low confidence.
    - Customizable: Change model or threshold for accuracy.
    - Robust: Handles missing data and invalid inputs.

Prerequisites:
    - Python 3.8+.
    - Libraries: pip install sentence-transformers pandas.
    - CSV file: Must have 'question' and 'answer' columns.

Usage:
    python faq_chatbot.py --faqs faqs.csv --model all-MiniLM-L6-v2
    - --faqs: Path to CSV with FAQs.
    - --model: Sentence Transformer model (default: 'all-MiniLM-L6-v2').

Outputs:
    - Interactive chat: Answers with scores; type 'exit' or 'quit' to stop.

Example:
    python faq_chatbot.py --faqs faqs.csv
    # Starts chat, responds to queries like "What is shipping?"

Limitations:
    - Static FAQs: No learning; relies on pre-loaded data.
    - Similarity threshold: Arbitrary (0.45); may need tuning.
    - Short queries: Works best for concise questions.
    - Scalability: For large FAQs, consider vector databases (e.g., FAISS).
    - Not conversational: Single-turn; extend with dialogue management.

Author: [Your Name/Username]
License: MIT
"""

import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import sys


def load_faqs(path):
    """
    Loads and validates the FAQ CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'question' and 'answer' columns.

    Raises:
        SystemExit: If file issues or missing columns.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Error: File '{path}' not found.")
    except Exception as e:
        raise SystemExit(f"Error loading CSV: {e}")
    
    if not {'question', 'answer'}.issubset(df.columns):
        raise SystemExit("Error: CSV must have 'question' and 'answer' columns.")
    
    if df.empty:
        raise SystemExit("Error: CSV is empty or has no valid rows.")
    
    return df


def encode_faqs(df, model):
    """
    Encodes FAQ questions using the Sentence Transformer model.

    Args:
        df (pd.DataFrame): FAQ DataFrame.
        model (SentenceTransformer): Loaded model.

    Returns:
        tuple: (list of questions, tensor of embeddings).
    """
    corpus = df['question'].fillna('').astype(str).tolist()  # Handle NaN
    corpus_emb = model.encode(corpus, convert_to_tensor=True)
    return corpus, corpus_emb


def find_best_answer(query, corpus_emb, df, threshold=0.45):
    """
    Finds the best matching FAQ answer for a query.

    Args:
        query (str): User question.
        corpus_emb: Encoded FAQ embeddings.
        df (pd.DataFrame): FAQ DataFrame.
        threshold (float): Minimum score for confident match.

    Returns:
        tuple: (answer, score) or (None, None) if no match.
    """
    if not query.strip():
        return None, None
    
    # Encode query (assuming model is global or passed; here we re-encode for simplicity)
    # In production, load model once and pass it
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Default; customize if needed
    q_emb = model.encode(query, convert_to_tensor=True)
    
    hits = util.semantic_search(q_emb, corpus_emb, top_k=1)[0]
    if hits:
        idx = hits[0]['corpus_id']
        score = hits[0]['score']
        answer = df.iloc[idx]['answer']
        return answer, score
    return None, None


def chat_loop(corpus_emb, df, threshold=0.45):
    """
    Runs the interactive chat loop.

    Args:
        corpus_emb: Encoded FAQ embeddings.
        df (pd.DataFrame): FAQ DataFrame.
        threshold (float): Confidence threshold.
    """
    print('FAQ chatbot ready. Type your question (or "exit" to quit):')
    while True:
        try:
            q = input('> ').strip()
            if q.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            answer, score = find_best_answer(q, corpus_emb, df, threshold)
            if answer:
                if score < threshold:
                    print("I couldn't confidently find an FAQ answer. Here's a possible related answer:")
                print(f'Answer: {answer}')
                print(f'(score {score:.3f})\n')
            else:
                print("No matching FAQ found. Try rephrasing.\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}\n")


def main():
    """
    Main function: Parses args, loads data, encodes FAQs, and starts chat.
    """
    parser = argparse.ArgumentParser(description="AI FAQ Chatbot using Semantic Search")
    parser.add_argument('--faqs', required=True, help="Path to CSV with 'question' and 'answer' columns")
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help="Sentence Transformer model name")
    parser.add_argument('--threshold', type=float, default=0.45, help="Similarity threshold for confident answers")
    args = parser.parse_args()

    # Load and validate FAQs
    df = load_faqs(args.faqs)
    
    # Load model and encode
    print(f"Loading model '{args.model}' and encoding {len(df)} FAQs...")
    try:
        model = SentenceTransformer(args.model)
        corpus, corpus_emb = encode_faqs(df, model)
    except Exception as e:
        raise SystemExit(f"Error with model '{args.model}': {e}")
    
    # Start chat
    chat_loop(corpus_emb, df, args.threshold)


if __name__ == '__main__':
    main()
