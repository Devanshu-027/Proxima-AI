# Medical Assistant: Rule-Based Diagnosis Tool - Full Detailed Workflow

## Introduction:

This document provides a detailed workflow of the Medical Assistant: Rule-Based Diagnosis Tool. It explains step-by-step processes, technical components, and interaction flows at a level suitable for thesis writing or comprehensive project documentation.

1. Project Overview

---

* The Medical Assistant suggests possible diseases based on input symptoms.
* It uses a rule-based system with a predefined symptom-disease mapping.
* Implemented in Python with a command-line interface.

2. System Architecture

---

* Modules:

  1. Argument Parsing
  2. Symptom Cleaning & Normalization
  3. Diagnosis Engine
  4. Output Formatting
* Libraries: Only built-in libraries (`argparse`, `collections`).

3. Data Loading and Mapping

---

* Knowledge base (`SYMPTOM_MAP`) is hardcoded.
* Maps symptoms to multiple diseases.
* Can be expanded for more accuracy.

4. Symptom Input Handling

---

* Accept user input via command-line:

  ```bash
  python medical_assistant.py --symptoms cough headache fever
  ```
* Each symptom is:

  * Stripped of whitespace
  * Lowercased
  * Spaces removed (e.g., 'body ache' -> 'bodyache')

5. Diagnosis Engine

---

* For each cleaned symptom:

  * Lookup matching diseases in `SYMPTOM_MAP`.
  * Increment disease score for each match.
* After processing all symptoms:

  * Sort diseases by descending score.
  * Return ranked list of `(disease, score)`.

6. Output Formatting

---

* Display ranked list of possible diagnoses.
* Example:

  ```
  flu (score=3)
  common cold (score=2)
  dengue (score=1)
  ```
* If no matches found:

  * Suggest expanding `SYMPTOM_MAP`.
  * Display educational disclaimer.

7. System Workflow Diagram

---

```
User Input (Symptoms) --> Clean & Normalize --> Match in SYMPTOM_MAP
       --> Count overlaps per disease --> Sort by score descending --> Display results
```

8. Error Handling

---

* Missing symptoms argument: print error and exit.
* Empty or invalid symptom strings: filtered out.
* Robust to unexpected inputs.

9. Configuration and Customization

---

* Expand `SYMPTOM_MAP` to include more symptoms/diseases.
* Integrate ML models to enhance accuracy.
* Modify scoring logic or thresholds as needed.

10. Limitations

---

* Rule-based only: no probabilistic reasoning.
* Covers common symptoms, not exhaustive.
* Not a substitute for professional medical advice.
* Accuracy depends on input quality and knowledge base coverage.

11. Future Enhancements

---

* Integrate ML classifiers trained on real datasets.
* Add web/GUI interface.
* Multi-language support for symptoms.
* Multi-symptom weighting and advanced pattern matching.

12. Deployment Steps

---

1. Place `medical_assistant.py` in project directory.

2. Open terminal/command prompt.

3. Run program with symptoms:

   ```bash
   python medical_assistant.py --symptoms cough fever headache
   ```

4. Review ranked disease output.

5. Conclusion

---

The Medical Assistant demonstrates rule-based diagnostic logic, symptom processing, scoring, ranking, and output presentation. It serves as a foundation for educational purposes and as a stepping stone toward A
