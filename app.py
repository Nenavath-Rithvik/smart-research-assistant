import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import io




st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 AI Research Companion: Summarize, Query & Practice")
st.markdown("""
This tool allows you to:
- 📌 Summarize long academic or research documents
- ❓ Ask custom questions based on content
- 🎯 Practice comprehension via 'Challenge Me' mode
""")

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose an action:")

st.sidebar.markdown("- 📄 Upload Document\n- 📌 Summary\n- 🧠 QA Mode\n- 🎯 Challenge Me\n\n---\n👨‍💻 Developed by [Your Name]")

uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])


def read_txt(file):
    return file.read().decode("utf-8")


def read_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        return extract_text(tmp_file.name)

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")


def answer_question(context, question):
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def generate_questions(text, num_questions=3):
    questions = set()
    inputs = f"generate questions: {text[:800]}"
    input_ids = qg_tokenizer.encode(inputs, return_tensors="pt")
    outputs = qg_model.generate(
        input_ids,
        max_length=64,
        num_return_sequences=10,
        do_sample=True,
        temperature=0.8
    )
    for output in outputs:
        question = qg_tokenizer.decode(output, skip_special_tokens=True).strip()
        if question.endswith("?"):
            questions.add(question)
        if len(questions) == num_questions:
            break
    return list(questions)



if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    st.write("Filename:", uploaded_file.name)

    if uploaded_file.type == "text/plain":
        full_text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        full_text = read_pdf(uploaded_file)
    else:
        full_text = ""

    with st.expander("📃 Full Text (Click to Expand)"):
        st.write(full_text[:3000])

    st.subheader("📌 Auto-Generated Summary:")
    with st.spinner("Summarizing..."):
        if len(full_text.strip()) < 100:
            st.warning("⚠️ The document content is too short for summarization or QA. Please upload a more detailed document.")
        else:
            st.subheader("📌 Auto-Generated Summary:")
            with st.spinner("Summarizing..."):
                summary = summarize_text(full_text)
                st.success("✅ Summary generated:")
                st.write(summary)

    st.subheader("🧠 Ask Anything from the Document")
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(full_text, question)
            st.success("✅ Answer:")
            st.write(answer)

    st.subheader("🧠 Challenge Me Mode")
    if st.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            if "user_answers" not in st.session_state:
                st.session_state.user_answers = []

            if "challenge_questions" not in st.session_state:
                st.session_state.challenge_questions = generate_questions(full_text)

        challenge_questions = st.session_state.challenge_questions
        st.subheader("🧠 Challenge Questions")

        for i, question in enumerate(challenge_questions):
            st.markdown(f"**Q{i+1}:** {question}")
            user_input = st.text_input(f"Your answer to Q{i+1}:", key=f"user_answer_{i}")

    if st.button("Evaluate Answers"):
        for i, (q, ua) in enumerate(user_answers):
            correct_ans = answer_question(full_text, q)
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"✅ Correct Answer: {correct_ans}")
            if ua.lower().strip() == correct_ans.lower().strip():
                st.success("Your answer is correct!")
            else:
                st.error("Your answer is incorrect.")
                st.markdown("📌 Justification: Answer derived from document content.")

        if st.button("📥 Download Results as CSV"):
            data = []
            for i, (q, ua) in enumerate(user_answers):
                correct_ans = answer_question(full_text, q)
                data.append({
                    "Question": q,
                    "Your Answer": ua,
                    "Correct Answer": correct_ans
                })
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="challenge_results.csv",
                mime="text/csv"
            )

        
