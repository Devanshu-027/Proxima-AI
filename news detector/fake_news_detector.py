"""
Fake News Detector using Zero-Shot Classification
==================================================

Description:
    This script uses AI (Hugging Face Transformers) for zero-shot text classification
    to detect fake news. It categorizes input text into predefined labels like
    'real news', 'fake news', 'opinion', or 'satire' with confidence scores.

Purpose:
    - Demonstrate zero-shot learning for text classification.
    - Provide a tool for quick content verification using NLP.
    - Showcase professional Python: AI integration, argument parsing, and output handling.
    - Educate on the limitations of AI in media analysis.

Key Features:
    - Zero-shot pipeline: Classifies without task-specific training.
    - Pre-trained model: Uses 'facebook/bart-large-mnli' for semantic understanding.
    - Customizable: Easy to change labels or model.
    - Scores: Probability distribution across labels.
    - Fast: Suitable for headlines or short texts.

Prerequisites:
    - Python 3.8+.
    - Libraries: pip install transformers.
    - Internet: For initial model download (cached afterward).

Usage:
    python fake_news_detector.py --text "Your headline here"
    - --text: The text to classify (e.g., a news headline).

Outputs:
    - Console: Input text and classification results with scores (0-1).

Example:
    python fake_news_detector.py --text "NASA FOUND WATER ON MOON"
    # Output: real news: 0.685, opinion: 0.121, etc.

Limitations:
    - Zero-shot: Relies on model generalization; may not handle domain-specific nuances.
    - Probabilistic: Scores are estimates, not certainties.
    - Bias: Model may reflect training data biases (e.g., toward certain news styles).
    - Not foolproof: Sarcasm or context can lead to errors.
    - For production: Combine with fact-checking APIs (e.g., Google Fact Check) or fine-tune on labeled data.

Author: [Your Name/Username]
License: MIT
"""

from transformers import pipeline
import argparse


def classify_text(text, model_name='facebook/bart-large-mnli', labels=None):
    """
    Classifies the input text using zero-shot classification.

    Args:
        text (str): The text to classify.
        model_name (str): Hugging Face model name (default: BART MNLI).
        labels (list): Candidate labels (default: ['real news', 'fake news', 'opinion', 'satire']).

    Returns:
        dict: Classification result with 'labels' and 'scores'.

    Raises:
        RuntimeError: If model loading or classification fails.
    """
    if labels is None:
        labels = ['real news', 'fake news', 'opinion', 'satire']
    
    try:
        classifier = pipeline('zero-shot-classification', model=model_name)
        result = classifier(text, candidate_labels=labels)
        return result
    except Exception as e:
        raise RuntimeError(f"Error in classification: {e}")


def main():
    """
    Main function: Parses args, validates input, runs classification, and prints results.
    """
    parser = argparse.ArgumentParser(description="AI Fake News Detector using Zero-Shot Classification")
    parser.add_argument('--text', required=True, help="Text to classify (e.g., a news headline)")
    parser.add_argument('--model', default='facebook/bart-large-mnli', help="Hugging Face model name")
    args = parser.parse_args()

    # Validate input
    if not args.text or not args.text.strip():
        print("Error: No text provided. Please supply valid text to classify.")
        return

    print('Input text:')
    print(args.text)
    
    try:
        result = classify_text(args.text, model_name=args.model)
        print('\nClassification results:')
        for lab, score in zip(result['labels'], result['scores']):
            print(f'{lab}: {score:.3f}')
        print('\nDisclaimer: This is an AI estimate. Verify with reliable sources.')
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
