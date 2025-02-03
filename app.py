import os
import json
import csv
import uuid
import pdfplumber
from io import StringIO
import streamlit as st
from predibase import Predibase, FinetuningConfig
from groq import Groq

# Optional: for DOCX support
try:
    import docx
except ImportError:
    st.error("Please install python-docx for DOCX file support (pip install python-docx)")

# ------------------------------
# Configuration and Initialization
# ------------------------------

# Initialize Predibase with your API token
pb = Predibase(api_token="pb_1w6VuPpVUsqASm2CSEjPeQ")

# Set your Groq API key and default model
GROQ_API_KEY = "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"  # Replace if necessary
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# ------------------------------
# Helper Functions
# ------------------------------

def groq_generate(prompt: str, model: str = MODEL_NAME, temperature: float = 0.7) -> str:
    """
    Uses Groq's LLM to generate a response for a given prompt.
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error communicating with Groq: {e}")
        return ""

def extract_text_from_pdf_file(pdf_file) -> str:
    """
    Extracts text from a PDF file using pdfplumber.
    """
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx_file(docx_file) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    """
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def generate_questions_agent(context: str, max_questions: int = 5) -> list:
    """
    Generates distinct and clear questions from the provided text using Groq.
    The prompt instructs the model to output exactly one question per line.
    """
    prompt_template = f"""Generate {max_questions} distinct and clear questions based solely on the text below.
Do not include any extra commentary or instructions.
List each question on a new line.

Text:
\"\"\"{context}\"\"\"

Questions:"""
    try:
        response_text = groq_generate(prompt_template, temperature=0.8)
        lines = response_text.strip().split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line[0] in "-*"):
                line = line.lstrip("0123456789. )-").strip()
            if line:
                questions.append(line)
        return questions[:max_questions]
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def generate_answer_agent(question: str, context: str) -> str:
    """
    Generates a clear and concise answer for a given question based on the text.
    The answer is expected to be one or two short paragraphs.
    """
    prompt_template = f"""Based on the text below, provide a clear and concise answer to the following question in one or two short paragraphs.

Text:
\"\"\"{context}\"\"\"

Question: {question}

Answer:"""
    try:
        answer = groq_generate(prompt_template, temperature=0.7)
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return ""

def json_to_predibase_csv(json_data, custom_uuid: str) -> StringIO:
    """
    Converts a list of Q&A dictionaries into a CSV formatted with a Predibase-style prompt.
    The CSV will include three columns: prompt, completion, and uuid.
    """
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(['prompt', 'completion', 'uuid'])
    for entry in json_data:
        question = entry.get('question', 'N/A')
        final_answer = entry.get('final_answer', 'N/A')
        prompt = f"""<s>[INST] The following question contains information about the EXPLORE app. Please answer based on the Question.

Question: {question}

Answer: [/INST]"""
        csv_writer.writerow([prompt, final_answer, custom_uuid])
    csv_buffer.seek(0)
    return csv_buffer

# ------------------------------
# Streamlit App
# ------------------------------

st.title("Predibase Fine-Tuning Pipeline")

# --- Step 1: File Upload and Q&A Generation ---
st.header("Step 1: Upload File and Generate CSV Dataset")
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
num_questions = st.number_input("Number of Questions to Generate", min_value=1, max_value=30, value=5, step=1)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        context_text = extract_text_from_pdf_file(uploaded_file)
    elif file_extension == "docx":
        context_text = extract_text_from_docx_file(uploaded_file)
    elif file_extension == "txt":
        context_text = uploaded_file.read().decode("utf-8")
    else:
        context_text = ""
    
    if context_text:
        st.success("Text successfully extracted from the uploaded file.")
    else:
        st.error("No text could be extracted from the file.")

    if st.button("Generate Q&A and Create CSV Dataset"):
        with st.spinner("Generating Q&A pairs..."):
            # First, generate questions using the provided number
            questions = generate_questions_agent(context_text, max_questions=num_questions)
            total = len(questions)
            qa_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            for idx, q in enumerate(questions, start=1):
                answer = generate_answer_agent(q, context_text)
                qa_list.append({
                    "question": q,
                    "final_answer": answer
                })
                progress_bar.progress(idx / total)
                status_text.text(f"Generated {idx}/{total} Q&A pairs")
            qa_data = qa_list
        custom_uuid = str(uuid.uuid4())
        csv_buffer = json_to_predibase_csv(qa_data, custom_uuid)
        csv_dataset = csv_buffer.getvalue()
        st.download_button(label="Download CSV Dataset", data=csv_dataset, file_name="predibase_dataset.csv", mime="text/csv")
        st.success("CSV dataset generated.")

# --- Step 2: Fine-Tuning ---
st.header("Step 2: Fine-Tune Model on Predibase")
dataset_name = st.text_input("Dataset Name", value="Enter Your dataset name")
uploaded_csv = st.file_uploader("Upload CSV for Fine-Tuning", type=["csv"], key="ft_csv")
epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5, step=1)
adapter_name = st.text_input("Adapter Name", value="Enter adaptor name")
repo_name = st.text_input("Repository Name", value="Enter adaptor name")
repo_desc = st.text_input("Repository Description", value="Enter repo description")

# Fixed parameters: rank = 16, learning_rate = 0.0002, temperature = 0.5, top_p = 0.1
if st.button("Start Fine-Tuning") and uploaded_csv is not None:
    # Save the uploaded CSV to a temporary file
    csv_file_path = "predibase_dataset_uploaded.csv"
    with open(csv_file_path, "wb") as f:
        f.write(uploaded_csv.read())
    
    try:
        dataset = pb.datasets.from_file(csv_file_path, name=dataset_name)
    except Exception as e:
        st.error(f"Error uploading dataset: {e}")
    
    repo = pb.repos.create(name=repo_name, description=repo_desc, exists_ok=True)
    try:
        adapter = pb.adapters.create(
            config=FinetuningConfig(
                base_model=adapter_name,
                epochs=epochs,
                rank=16,
                learning_rate=0.0002,
                target_modules=["q_proj", "v_proj", "k_proj"],
            ),
            dataset=dataset,
            repo=repo,
            description="Fine-tuning on Hyundai IONIQ 5 Q&A data"
        )
        st.success("Successfully requested finetuning of llama-3-1-8b-instruct")
        st.info("Your finetuning is completed!")
    except Exception as e:
        st.error(f"Error during fine-tuning: {e}")
