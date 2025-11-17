# Resume Screener using Sentence Transformers - Full Detailed Workflow

## Introduction:

This document provides a comprehensive workflow for the AI Resume Screener project, which automatically evaluates resumes against a job description using semantic similarity.

1. Project Overview

---

* Purpose: Automate resume screening using NLP and AI.
* Type: AI-powered semantic similarity tool.
* Input: Job description (.txt) and multiple resumes (.txt).
* Output: Ranked table of resumes, CSV, and a bar chart.

2. System Architecture

---

* Modules:

  1. Argument Parsing: Accepts job description file, resume folder, and optional model.
  2. File Handling: Reads and validates .txt files.
  3. Model Loading: Loads Sentence Transformer model.
  4. Encoding: Converts text to embeddings.
  5. Similarity Calculation: Computes cosine similarity.
  6. Ranking: Sorts resumes by score.
  7. Output: Tabulated results, CSV, bar chart.
* Libraries: sentence-transformers, pandas, matplotlib, tabulate, argparse, os, glob.
* Optional GPU support for faster encoding.

3. Input Handling

---

* Job description (.txt) must exist and contain valid text.
* Resumes must be in a specified folder as .txt files.
* Validation checks ensure files/folders exist.

4. Semantic Similarity Engine

---

* Uses Sentence Transformers to encode job description and resumes.
* Computes cosine similarity between job description embedding and each resume embedding.
* Higher similarity score indicates better match.

5. Output Handling

---

* Displays a ranked table with Resume names and Similarity Scores.
* Exports results to 'resume_screener_results.csv'.
* Generates horizontal bar chart 'resume_screener_chart.png' for visualization.

6. Workflow Diagram

---

```
User Input (JD + Resumes) --> Validate Files --> Load Model --> Encode Texts
       --> Compute Cosine Similarities --> Rank Results --> Display Table & Save CSV & Plot Chart
```

7. Error Handling

---

* Checks for missing or empty files.
* Skips invalid resumes and logs warnings.
* Handles model loading errors gracefully.

8. Technical Components

---

* Python 3.8+
* Libraries: sentence-transformers, pandas, matplotlib, tabulate
* Algorithm: Semantic similarity via cosine similarity on embeddings
* Extensibility: Switch models or add preprocessing.

9. Limitations

---

* Only supports plain text files; no PDFs or Word documents.
* Semantic similarity may not capture domain-specific jargon.
* Not a substitute for human evaluation.
* Accuracy depends on model quality; larger models provide better results but slower.

10. Future Enhancements

---

* Add PDF/Word parsing.
* Integrate NLP preprocessing (stopwords removal, lemmatization).
* Batch processing for large resume sets.
* Web or GUI interface.
* Integrate ML-based ranking for multi-factor scoring.

11. Deployment Steps

---

1. Install Python 3.8+.

2. Install libraries: pip install sentence-transformers pandas matplotlib tabulate.

3. Prepare job description and resumes folder.

4. Run the script:

   ```bash
   python resume_screener.py --jd job_desc.txt --resumes resumes/
   ```

5. Review console output, CSV, and bar chart.

6. Conclusion

---

This project demonstrates AI-based semantic resume screening, combining NLP embeddings, cosine similarity, and data visualization for automated candidate shortlisting.
