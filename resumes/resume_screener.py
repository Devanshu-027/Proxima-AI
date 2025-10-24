"""
Resume Screener using Sentence Transformers
===========================================

Description:
    This AI-powered resume screener compares a job description with multiple resumes
    using semantic similarity (via Sentence Transformers and cosine similarity).
    It ranks resumes by relevance, displays results in a table, saves to CSV,
    and generates a bar chart for visualization.

Purpose:
    - Automate resume screening for recruiters using NLP/AI.
    - Demonstrate semantic search and similarity in real-world applications.
    - Showcase professional Python: file handling, AI integration, data visualization.

Key Features:
    - Semantic similarity: Understands meaning, not just keywords.
    - Ranked results: Sorted by similarity score (0-1, higher is better).
    - Tabulated display: Clean grid output.
    - CSV export: For further analysis.
    - Bar chart: Horizontal plot with scores.
    - Customizable models: Switch to better/finer models for accuracy.

Prerequisites:
    - Python 3.8+.
    - Libraries: pip install sentence-transformers pandas matplotlib tabulate.
    - Files: Job description as .txt, resumes as .txt in a folder.
    - GPU: Optional, speeds up encoding.

Usage:
    python resume_screener.py --jd job_desc.txt --resumes resumes/ --model all-MiniLM-L6-v2
    - --jd: Path to job description text file.
    - --resumes: Folder containing resume .txt files.
    - --model: Sentence Transformer model (default: 'all-MiniLM-L6-v2').

Outputs:
    - Console: Ranked table of resumes with scores.
    - CSV: 'resume_screener_results.csv' with Resume and Similarity Score columns.
    - Chart: 'resume_screener_chart.png' (horizontal bar chart).

Example:
    python resume_screener.py --jd job_desc.txt --resumes resumes/
    # Loads model, compares, displays table, saves files.

Limitations:
    - Requires plain text files; no PDF/Word support.
    - Similarity is semantic; may not catch domain-specific jargon perfectly.
    - Model-dependent: Larger models (e.g., 'all-mpnet-base-v2') are more accurate but slower.
    - Not a replacement for human review; use for shortlisting.

Author: [Your Name/Username]
License: MIT
"""

import argparse
import os
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate


def read_text(path):
    """
    Reads and returns text content from a file with error handling.

    Args:
        path (str): Path to the text file.

    Returns:
        str: File content.

    Raises:
        FileNotFoundError: If file doesn't exist.
        IOError: If reading fails.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except IOError as e:
        raise IOError(f"Error reading file {path}: {e}")


def main():
    """
    Main function: Parses args, loads model, processes files, computes similarities,
    ranks results, displays table, saves CSV, and plots chart.
    """
    # Argument parser with validation
    parser = argparse.ArgumentParser(description="AI Resume Screener using Sentence Transformers")
    parser.add_argument('--jd', required=True, help="Path to job description text file")
    parser.add_argument('--resumes', required=True, help="Folder containing candidate resumes (.txt)")
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help="Sentence Transformer model name")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.jd):
        raise FileNotFoundError(f"Job description file not found: {args.jd}")
    if not os.path.isdir(args.resumes):
        raise NotADirectoryError(f"Resumes folder not found: {args.resumes}")

    print(f"🔍 Loading model: {args.model} ...")
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        raise RuntimeError(f"Error loading model '{args.model}': {e}")

    print(f"📄 Reading Job Description: {args.jd}")
    jd_text = read_text(args.jd)
    if not jd_text.strip():
        raise ValueError("Job description file is empty.")
    jd_emb = model.encode(jd_text, convert_to_tensor=True)

    resumes = glob.glob(os.path.join(args.resumes, '*.txt'))
    if not resumes:
        print("❌ No .txt resumes found in the specified folder!")
        return

    print(f"🧠 Comparing {len(resumes)} resumes with the Job Description...\n")
    results = []

    # Calculate similarity for each resume
    for resume_path in resumes:
        try:
            resume_text = read_text(resume_path)
            if not resume_text.strip():
                print(f"⚠️  Skipping empty resume: {os.path.basename(resume_path)}")
                continue
            resume_emb = model.encode(resume_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(jd_emb, resume_emb).item()
            results.append({
                'Resume': os.path.basename(resume_path),
                'Similarity Score': round(similarity, 4)
            })
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(resume_path)}: {e}. Skipping.")

    if not results:
        print("❌ No valid resumes processed!")
        return

    # Sort results by similarity score descending
    results = sorted(results, key=lambda x: x['Similarity Score'], reverse=True)

    # Display tabulated results
    print("📊 Ranked Resume Results:")
    print(tabulate(results, headers='keys', tablefmt='grid'))

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'resume_screener_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to {csv_path}")

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.barh([r['Resume'] for r in results], [r['Similarity Score'] for r in results], color='skyblue')
    plt.xlabel('Similarity Score')
    plt.ylabel('Resume')
    plt.title('Resume Similarity to Job Description')
    plt.gca().invert_yaxis()  # Highest score on top
    plt.tight_layout()
    chart_path = 'resume_screener_chart.png'
    plt.savefig(chart_path)
    print(f"📈 Bar chart saved to {chart_path}")
    plt.show()


if __name__ == "__main__":
    main()
