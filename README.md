#  Smart Research Assistant 

An AI-powered Streamlit web app that helps users summarize research documents, ask questions based on content, and test their understanding through quiz-like challenges — all using cutting-edge NLP models.

---

##  Features

-  **Upload PDF or TXT** research papers or notes.
-  **Auto-Summarization** using `DistilBART`.
-  **Ask Questions** from the document using `DistilBERT`.
-  **Challenge Me Mode**: Auto-generates comprehension questions using `T5`, and evaluates your answers.

---

## 🔧 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- `pdfminer.six`
- `T5`, `DistilBART`, `DistilBERT` models

---

##  Installation

### Clone the repository

```bash
git clone https://github.com/Nenavath-Rithvik/smart-research-assistant.git
cd smart-research-assistant

Create a virtual environment :
python -m venv assistant_env
assistant_env\Scripts\activate  # Windows

Install dependencies:
pip install -r requirements.txt

Run the App:
streamlit run app.py
Open http://localhost:8501 in your browser to use the app.

Folder Structure:
smart-research-assistant/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # List of required Python packages
└── README.md            # Project documentation (this file)


Models Used:
Summarization: sshleifer/distilbart-cnn-12-6

Question Answering: distilbert-base-uncased-distilled-squad

Question Generation: mrm8488/t5-base-finetuned-question-generation-ap

 Author:
RITHVIK NENAVATH
