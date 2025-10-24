"""
Sentiment Analyzer using Transformers
=====================================

Description:
    This script performs AI-powered sentiment analysis on a CSV file of reviews.
    It classifies each review as positive/negative (or custom sentiments), adds
    labels to the data, saves an updated CSV, and generates a bar chart of
    sentiment distribution.

Purpose:
    - Demonstrate sentiment analysis with pre-trained NLP models.
    - Provide a tool for batch processing of text data.
    - Showcase professional Python: data handling, AI pipelines, visualization.
    - Educate on model selection and limitations in sentiment tasks.

Key Features:
    - Pre-trained models: Uses DistilBERT by default; customizable.
    - Batch processing: Handles multiple reviews with progress bar.
    - Data export: Updated CSV with sentiment column.
    - Visualization: Bar chart of sentiment counts.
    - Robustness: Handles missing data, truncates long texts, and skips errors.
    - Summary: Counts and percentages for insights.

Prerequisites:
    - Python 3.8+.
    - Libraries: pip install transformers pandas matplotlib tqdm.
    - CSV file: Must have a 'review' column with text data.

Usage:
    python sentiment_analyzer.py --file reviews.csv --model distilbert-base-uncased-finetuned-sst-2-english
    - --file: Path to CSV file.
    - --model: Optional Hugging Face model name.

Outputs:
    - Updated CSV: '{original}_with_sentiment.csv' with added 'sentiment' column.
    - Chart: 'sentiment_distribution.png' (bar plot).
    - Console: Progress, summary stats, and messages.

Example:
    python sentiment_analyzer.py --file reviews.csv
    # Processes reviews, saves files, prints summary.

Limitations:
    - Model bias: Pre-trained on general data; may not fit specific domains.
    - Text limits: Truncates to 512 chars; long reviews may lose context.
    - Binary by default: Some models are positive/negative only.
    - Not real-time: Batch processing; for live use, integrate with APIs.
    - Accuracy: Improve with fine-tuning on labeled data.

Author: [Your Name/Username]
License: MIT
"""

import argparse
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar


def load_and_validate_csv(file_path):
    """
    Loads and validates the CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        SystemExit: If file issues or missing 'review' column.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise SystemExit(f"Error: File '{file_path}' not found.")
    except Exception as e:
        raise SystemExit(f"Error loading CSV: {e}")
    
    if 'review' not in df.columns:
        raise SystemExit("Error: CSV must contain a 'review' column.")
    
    return df


def analyze_sentiments(reviews, model_name):
    """
    Analyzes sentiments for a list of reviews.

    Args:
        reviews (list): List of review texts.
        model_name (str): Hugging Face model name.

    Returns:
        list: List of sentiment labels.
    """
    try:
        classifier = pipeline('sentiment-analysis', model=model_name)
    except Exception as e:
        raise SystemExit(f"Error loading model '{model_name}': {e}")
    
    sentiments = []
    for text in tqdm(reviews, desc="Analyzing sentiments"):
        try:
            # Truncate to 512 chars (model limit) and classify
            res = classifier(text[:512])[0]
            sentiments.append(res['label'])
        except Exception as e:
            print(f"Warning: Failed to analyze review '{text[:50]}...': {e}. Skipping.")
            sentiments.append('UNKNOWN')  # Fallback for errors
    return sentiments


def save_results_and_plot(df, original_file):
    """
    Saves updated CSV and generates/saves sentiment distribution plot.

    Args:
        df (pd.DataFrame): DataFrame with 'sentiment' column.
        original_file (str): Original CSV file path.
    """
    # Save updated CSV
    output_csv = original_file.replace('.csv', '_with_sentiment.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV with sentiments to '{output_csv}'.")
    
    # Generate and save plot
    counts = df['sentiment'].value_counts()
    counts.plot(kind='bar', color=['green', 'red', 'blue', 'orange'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=150)
    print("Saved sentiment distribution chart to 'sentiment_distribution.png'.")
    
    # Print summary
    total = len(df)
    print("\nSentiment Summary:")
    for label, count in counts.items():
        percentage = (count / total) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")


def main():
    """
    Main function: Parses args, loads data, analyzes sentiments, saves results, and plots.
    """
    parser = argparse.ArgumentParser(description="AI Sentiment Analyzer for CSV reviews")
    parser.add_argument('--file', required=True, help="Path to CSV file with 'review' column")
    parser.add_argument('--model', default='distilbert-base-uncased-finetuned-sst-2-english', 
                        help="Hugging Face model name (default: DistilBERT for sentiment)")
    args = parser.parse_args()
    
    # Load and validate CSV
    df = load_and_validate_csv(args.file)
    
    # Clean and prepare reviews
    reviews = df['review'].fillna('').astype(str).tolist()  # Handle NaN/missing as empty strings
    if not reviews:
        raise SystemExit("Error: No valid reviews found in the CSV.")
    
    print(f"Loaded {len(reviews)} reviews from '{args.file}'. Analyzing with model: {args.model}...")
    
    # Analyze sentiments
    sentiments = analyze_sentiments(reviews, args.model)
    
    # Add to DataFrame
    df['sentiment'] = sentiments
    
    # Save results and plot
    save_results_and_plot(df, args.file)


if __name__ == '__main__':
    main()
