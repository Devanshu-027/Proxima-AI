# FAQ Chatbot using Semantic Search - Full Detailed Workflow

## Introduction:

This document provides a detailed workflow of the AI-powered FAQ Chatbot using semantic search. It outlines the step-by-step processes, technical components, and interaction flows in a level of detail sufficient for academic research, thesis writing, or comprehensive project documentation.

1. **Project Overview**

---

* The FAQ Chatbot is designed to answer user queries by finding semantically similar FAQs.
* It uses Sentence Transformers to encode questions and perform semantic similarity search.
* The chatbot is implemented in Python and interacts via a command-line interface.

2. **System Architecture**

---

* **Modules:**

  1. Data Loading
  2. Embedding / Encoding
  3. Semantic Search Engine
  4. Query Processing and Matching
  5. Interactive Chat Loop
* **Libraries Used:**

  * pandas (data handling)
  * numpy (numerical operations)
  * sentence-transformers (embedding and semantic similarity)
  * argparse (command-line arguments)

3. **Data Loading and Validation**

---

* CSV file is required with columns `question` and `answer`.
* Steps:

  1. Load CSV using `pandas.read_csv()`.
  2. Validate presence of required columns.
  3. Check for empty rows or missing data.
  4. Raise descriptive errors for missing file or incorrect format.

4. **Encoding FAQ Questions**

---

* Sentence Transformer model converts text questions into high-dimensional vector embeddings.
* Steps:

  1. Initialize model (default: 'all-MiniLM-L6-v2').
  2. Clean and standardize text data (convert NaN to empty strings).
  3. Generate embeddings for each FAQ question.
  4. Store embeddings for similarity comparison.
* Output:

  * List of questions
  * Tensor of embeddings

5. **Query Processing and Similarity Search**

---

* User input is received via command-line interface.
* Steps:

  1. Accept user query input.
  2. Encode query using the same Sentence Transformer model.
  3. Compute cosine similarity with all FAQ embeddings using `util.semantic_search()`.
  4. Retrieve top match(es) and similarity scores.
  5. Compare score with threshold (default: 0.45).

     * If score >= threshold: return answer.
     * If score < threshold: return answer with low-confidence warning.
     * If no match: prompt user to rephrase.

6. **Interactive Chat Loop**

---

* Provides a real-time conversation interface.
* Steps:

  1. Print greeting and instructions.
  2. Loop to accept user input continuously.
  3. Check for exit commands (`exit` or `quit`).
  4. Process query using the semantic search function.
  5. Handle exceptions and invalid inputs gracefully.
  6. Display answer and similarity score.
* Ensures robust handling of keyboard interrupts and unexpected errors.

7. **System Workflow Diagram (Logical Flow)**

---

```
CSV FAQ File --> Load & Validate --> Encode Questions --> Store Embeddings
                             ^
                             |
User Query --> Encode Query --> Semantic Search --> Score Check --> Return Answer
```

8. **Configuration and Customization**

---

* Users can specify:

  * CSV file path (`--faqs`)
  * Model (`--model`)
  * Similarity threshold (`--threshold`)
* Configurations allow:

  * Changing Sentence Transformer models
  * Adjusting confidence sensitivity
  * Supporting different domains of FAQ data

9. **Limitations**

---

* Static FAQs: no learning beyond CSV data.
* Single-turn queries only: no multi-turn conversational memory.
* Similarity threshold may require tuning.
* Scalability: for very large datasets, vector databases like FAISS recommended.
* Works best with concise, clear queries.

10. **Outputs and Logging**

---

* Interactive chat displays:

  * Answer text
  * Similarity score
* Errors and exceptions logged to console.
* Optional: developer can extend to file-based logging for audits.

11. **Future Enhancements**

---

* Multi-turn conversation memory
* Integration with FAISS or Milvus for large datasets
* Web or GUI-based front-end
* REST API deployment for multiple clients
* Persistent learning from user interactions
* Advanced ranking and context handling

12. **Deployment Steps**

---

1. Clone repository

2. Create Python virtual environment

3. Install required libraries

4. Place `faqs.csv` in project directory

5. Run `python faq_chatbot.py --faqs faqs.csv`

6. **Conclusion**

---

The FAQ Chatbot project demonstrates a structured approach to building an AI system for semantic question answering. Its workflow, from data loading to query processing and semantic matching, is modular, robust, and extendable. It can serve as a foundation for larger conversational AI projects, vector-based search systems, and practical applications in customer support, education, and research.
