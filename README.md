Proxima AI
==========

Overview
--------
Proxima AI is a collection of **7 practical AI projects in Python**. These projects span NLP, finance, time-series forecasting, resume screening, and conversational AI. Each project demonstrates real-world AI applications and includes fully functional scripts, visualizations, and outputs. The repository is ideal for learning, experimentation, and portfolio-building.

Projects
--------
1. **Resume Screener**
   - **Location:** resumes/resume_screener.py
   - **Description:** AI-powered tool that ranks resumes based on a job description using semantic similarity.
   - **Features:**
     - Semantic similarity with Sentence Transformers.
     - Ranked results in tabular form.
     - CSV export of scores.
     - Bar chart visualization.
   - **Usage Example:**
     ```
     python resumes/resume_screener.py --jd job_desc.txt --resumes resumes_folder/
     ```

2. **Sentiment Analyzer**
   - **Location:** sentiments/sentiment_analyzer.py
   - **Description:** Performs batch sentiment analysis on CSV reviews using a pre-trained NLP model.
   - **Features:**
     - Supports Hugging Face transformer models (default: DistilBERT SST-2).
     - Adds sentiment labels to the CSV.
     - Bar chart visualization of sentiment distribution.
   - **Usage Example:**
     ```
     python sentiments/sentiment_analyzer.py --file reviews.csv
     ```

3. **Fake News Detector**
   - **Location:** news_detector/
   - **Description:** Detects fake news articles or text inputs using an AI classifier.
   - **Features:**
     - NLP-based fake news classification.
     - Outputs prediction labels (REAL or FAKE).
   - **Usage Example:**
     ```
     python news_detector/fake_news_detector.py --input news_text.txt
     ```

4. **FAQ Chatbot**
   - **Location:** faq_chatbot/
   - **Description:** Conversational AI bot that answers questions based on a predefined FAQ dataset.
   - **Features:**
     - Semantic matching of questions.
     - Handles multiple topics or domains.
     - Can be extended with custom datasets.
   - **Usage Example:**
     ```
     python faq_chatbot/chatbot.py
     ```

5. **Stock Portfolio Optimizer**
   - **Location:** stock_portfolio/portfolio_optimizer.py
   - **Description:** Optimizes a stock portfolio to maximize the Sharpe ratio using Modern Portfolio Theory.
   - **Features:**
     - Fetches historical stock prices via Yahoo Finance.
     - Calculates returns, covariance, and risk-adjusted metrics.
     - Supports interactive input and Excel export.
   - **Usage Example:**
     ```
     python stock_portfolio/portfolio_optimizer.py --tickers "AAPL MSFT" --period 1y
     ```

6. **Stock Price Predictor**
   - **Location:** stock_predictor/stock_predictor.py
   - **Description:** Predicts future stock closing prices using an LSTM neural network.
   - **Features:**
     - Fetches historical stock data.
     - Preprocesses data with MinMax scaling.
     - Trains LSTM with dropout and early stopping.
     - Generates plots comparing actual vs. predicted prices.
   - **Usage Example:**
     ```
     python stock_predictor/stock_predictor.py --ticker AAPL --period 2y --window 60
     ```

7. **Medical Assistant (AI Chatbot)**
   - **Location:** medical_assistant.py
   - **Description:** A medical domain chatbot that answers queries and provides basic guidance.
   - **Features:**
     - NLP-based understanding of medical questions.
     - Conversational interface.
     - Can be extended with custom medical datasets.
   - **Usage Example:**
     ```
     python medical_assistant.py
     ```

Getting Started
---------------
Follow these steps to set up and run Proxima AI locally.

### 1. Clone the repository
git clone https://github.com/yourusername/proxima-ai.git
cd proxima-ai

shell
Copy code

### 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

shell
Copy code

### 3. Install required packages
pip install -r requirements.txt

sql
Copy code
*Note: Each project may have specific dependencies. See the respective folder for details.*

### 4. Run a project
Navigate to the project folder and execute the script as described in the “Projects” section. Example:
python resumes/resume_screener.py --jd job_desc.txt --resumes resumes_folder/

makefile
Copy code

### 5. Optional: Jupyter Notebook
Some projects may include notebooks for visualization. You can launch:
jupyter notebook

markdown
Copy code
and open the corresponding `.ipynb` files.

Requirements
------------
- Python 3.8 or higher
- Packages (included in `requirements.txt` below):
  - `transformers`
  - `sentence-transformers`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow`
  - `torch`
  - `yfinance`
  - `flask`
  - `streamlit`

Contributing
------------
Contributions are welcome! You can:
- Report bugs or issues.
- Suggest new AI projects.
- Submit pull requests with improvements or new features.

Please ensure code style consistency and add comments for clarity.

License
-------
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
-------
- **Developer:** Devanshu Kumar
- **Email:** your_email@example.com
- **GitHub:** https://github.com/yourusername

Acknowledgements
----------------
- Hugging Face Transformers
- Yahoo Finance API
- OpenAI API (for chatbot inspiration)
- Python community for open-source support
