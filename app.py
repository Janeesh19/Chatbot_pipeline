import os
import json
import csv
import uuid
import pdfplumber
from io import StringIO
import streamlit as st
from predibase import Predibase, FinetuningConfig
from groq import Groq
import re
from typing import List, Dict

# Optional: for DOCX support
try:
    import docx
except ImportError:
    st.error("Please install python-docx for DOCX file support (pip install python-docx)")

# ------------------------------
# Qdrant and WordLlama Initialization
# ------------------------------
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct, Distance
except ImportError:
    st.error("Please install qdrant-client (pip install qdrant-client)")
try:
    from wordllama import WordLlama
except ImportError:
    st.error("Please install wordllama for generating embeddings (pip install wordllama)")

import numpy as np

def flatten_embedding(embedding_vector):
    """
    Ensures that the embedding_vector is a flat list.
    If the vector is wrapped as a single-element list, extract and flatten that element.
    """
    if isinstance(embedding_vector, np.ndarray):
        embedding_vector = embedding_vector.tolist()
    if isinstance(embedding_vector, list) and len(embedding_vector) == 1:
        first = embedding_vector[0]
        if isinstance(first, (list, np.ndarray)):
            embedding_vector = np.array(first).flatten().tolist()
            return embedding_vector
    if isinstance(embedding_vector, list) and embedding_vector and isinstance(embedding_vector[0], (list, np.ndarray)):
        embedding_vector = np.array(embedding_vector).flatten().tolist()
    return embedding_vector

# Initialize Qdrant client (assumes Qdrant is running on localhost)
qdrant_client = QdrantClient(host="127.0.0.1", port=6333)
qdrant_collection = "qna_collection"
embedding_dimension = 256  # Adjust based on your model’s output dimension

collections = [coll.name for coll in qdrant_client.get_collections().collections]
if qdrant_collection not in collections:
    qdrant_client.recreate_collection(
        collection_name=qdrant_collection,
        vectors_config={"size": embedding_dimension, "distance": Distance.COSINE}
    )

wl = WordLlama.load()

# ------------------------------
# Configuration and Initialisation
# ------------------------------
pb = Predibase(api_token=st.secrets["general"]["OPENAI_API_KEY"])
GROQ_API_KEY = st.secrets["general"]["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# Fixed loss system prompt (hardcoded, not editable)
FIXED_LOSS_SYSTEM_PROMPT = (
    "Evaluate the following meta prompt for generating an abstract chain-of-thought (ACoT) explanation. "
    "The meta prompt instructs the model to first generate a generic explanation that describes the general approach to solving a problem. "
    "It should include:\n\n"
    "A brief overview of the type of problem or situation.\n"
    "The generic steps that one should follow to solve this kind of problem.\n"
    "An explanation of why each step is important.\n\n"
    "After outlining this generic procedure, the prompt then instructs the model to apply these steps to the specific problem provided.\n\n"
    "Please assess whether this meta prompt is clear, concise, and sufficiently instructive to ensure that the generated chain-of-thought:\n"
    "- Explains the general methodology in an abstract way,\n"
    "- Demonstrates the process step-by-step,\n"
    "- And then successfully applies that process to the specific question.\n"
    "Provide feedback on any ambiguities, missing instructions, or potential improvements to ensure that the ACoT fully teaches the procedure rather than simply providing the final answer."
)

# Fixed role description (hardcoded, not editable)
FIXED_ROLE_DESCRIPTION = (
    "You are a knowledgeable and patient instructor and tutor whose role is to teach problem-solving methods "
    "through an abstract chain-of-thought explanation. Your task is to first provide a clear, concise overview "
    "of the general approach to solving similar problems, detailing the key steps and the rationale behind each step. "
    "Then, you apply this general procedure to the specific problem at hand. Your explanation should be instructive and accessible, "
    "ensuring that anyone can learn the underlying methodology rather than just receiving the final answer."
)

# Initialize session state for optimized prompt if not already present
if "optimized_cot_prompt" not in st.session_state:
    st.session_state.optimized_cot_prompt = None

selected_step = st.sidebar.radio("Select a step:", 
                                 ["Upload & Q&A Generation", "Fine-Tuning Job", "Query Embedding Database"])

# ------------------------------
# COT Prompt Optimization Settings (Input for initial prompt only)
# ------------------------------
with st.expander("COT Prompt Optimization Settings"):
    initial_prompt_input = st.text_area(
        "Enter Initial Prompt for COT Optimization:",
        value="""You are a helpful instructor and tutor. Instead of jumping straight to the final answer, first generate an abstract chain-of-thought (ACoT) that explains the general approach to solving a problem. Your abstract explanation should include: 1. A brief overview of the type of problem or situation. 2. The generic steps that one should follow to solve this kind of problem. 3. An explanation of why each step is important. After outlining this generic procedure, apply these steps to the specific problem provided. For example, for a math word problem like 'There are 5 apples, John takes 2, how many remain?', your abstract explanation might be: 'This is a subtraction problem. First, identify the total number of items (apples). Then, determine how many items are removed. Finally, subtract to get the remaining number.' Now, apply that reasoning to the specific problem."""
    )
    if st.button("Optimize COT Prompt"):
        os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
        try:
            import textgrad as tg
        except ImportError:
            st.error("Please install textgrad (pip install textgrad)")
        tg.set_backward_engine(tg.get_engine("gpt-4o"), override=True)
        solution = tg.Variable(
            initial_prompt_input,
            requires_grad=True,
            role_description=FIXED_ROLE_DESCRIPTION
        )
        loss_system = tg.Variable(
            FIXED_LOSS_SYSTEM_PROMPT,
            requires_grad=False,
            role_description="System feedback prompt for English language optimization"
        )
        loss_fn = tg.TextLoss(loss_system)
        optimizer = tg.TGD([solution])
        loss = loss_fn(solution)
        loss.backward()
        optimizer.step()
        optimized_prompt = solution.value
        st.session_state.optimized_cot_prompt = optimized_prompt
        st.write("Optimized COT Prompt has been set:")
        st.markdown(optimized_prompt)  # Display as a full paragraph with preserved line breaks

# ------------------------------
# Helper Functions: File Extraction
# ------------------------------
def extract_text_from_pdf_file(pdf_file) -> str:
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
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# ------------------------------
# Helper Functions: Q&A Generation with CoT & Reflection
# ------------------------------
def groq_generate(prompt: str, model: str = MODEL_NAME, temperature: float = 0.7) -> str:
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed and accurate responses."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error communicating with Groq: {e}")
        return ""

def summarize_pdf_content(context: str) -> str:
    prompt = f"""Summarise the following content into a concise version highlighting the key points relevant for sales insights.

Content:
\"\"\"{context}\"\"\" 

Summary:"""
    return groq_generate(prompt, temperature=0.5)

def generate_questions_agent(context: str, max_questions: int = 5) -> List[str]:
    prompt_template = f"""You are an AI assistant tasked with generating clear, concise, and insightful questions based on the uploaded content.
Your goal is to create questions that will help sales teams effectively understand and present the key information from the content.

Content:
\"\"\"{context}\"\"\" 

Generate exactly {max_questions} sales-focused questions. These questions should:
- Uncover key benefits and differentiators.
- Address customer pain points and objections.
- Highlight pricing, features, and competitive advantages.
- Encourage engagement and conversation with potential buyers.

- When generating a question, include only the question itself (do not add any extra headings).
- Keep questions as concise as possible.
- When generating the final answer, do not mention chain-of-thought (CoT) or reflection in your answer.
  
Sales-optimized questions:"""
    try:
        response_text = groq_generate(prompt_template, temperature=0.8)
        raw_lines = response_text.strip().split('\n')
        questions = [line.strip('-•1234567890. ').strip() for line in raw_lines if line.strip().endswith('?')]
        return questions[:max_questions]
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def generate_cot_agent(question: str, context: str) -> str:
    """
    Generates an abstract chain-of-thought (ACoT) explanation.
    The prompt instructs the model to first explain a generic approach that applies to similar problems,
    then to apply that reasoning to the specific question.
    It uses the optimized prompt if available.
    """
    if st.session_state.optimized_cot_prompt:
        base_prompt = st.session_state.optimized_cot_prompt
    else:
        base_prompt = (
            "You are a helpful instructor and tutor. Instead of jumping straight to the final answer, "
            "first generate an abstract chain-of-thought (ACoT) that explains the general approach to solve such problems. "
            "Your abstract explanation should include: 1. A brief overview of the type of problem or situation. "
            "2. The generic steps that one should follow to solve this kind of problem. 3. An explanation of why each step is important. "
            "After presenting the abstract chain-of-thought, apply these steps to the specific problem below. "
            "For example, for a math word problem like 'There are 5 apples, John takes 2, how many remain?', your abstract explanation might be: "
            "'This is a subtraction problem. First, identify the total number of items (apples). Then, determine how many items are removed. Finally, subtract to get the remaining number.' "
            "Now, apply that reasoning to the specific problem."
        )
    prompt_template = f"""{base_prompt}

Context:
{context}

Question:
{question}

Abstract Chain-of-Thought (explain the procedure generically and then apply it to this problem):"""
    return groq_generate(prompt_template, temperature=0.7)

def reflection_agent(question: str, context: str, chain_of_thought: str) -> str:
    prompt_template = f"""You are a meticulous reviewer.

Please follow these steps:

1. Review the abstract chain-of-thought reasoning for accuracy and completeness.
2. Identify any errors, omissions, or areas for improvement.
3. Provide a corrected and improved chain-of-thought reasoning.
4. Ensure the final answer is clear and accurate.

Context:
{context}

Question:
{question}

Initial abstract chain-of-thought reasoning:
{chain_of_thought}

Reflection:"""
    return groq_generate(prompt_template, temperature=0.6)

def extract_final_answer_agent(reflection: str) -> str:
    prompt = f"""Based on the reflection below, provide the final, refined answer.

Reflection:
{reflection}

Please follow the instructions:
- Extract the final answer from the reflection.
- The final answer should be as concise as possible, but if more detail is needed, provide a clear explanation.
- Do not mention chain-of-thought (CoT) or reflection in your answer.
  
Answer:"""
    return groq_generate(prompt, temperature=0.5)

def process_question(question: str, context: str) -> Dict:
    chain_of_thought = generate_cot_agent(question, context)
    reflection = reflection_agent(question, context, chain_of_thought)
    final_answer = extract_final_answer_agent(reflection)
    return {
        'question': question,
        'generic_cot': chain_of_thought,
        'final_answer': final_answer
    }

def process_context(context: str, max_questions: int = 5) -> List[Dict]:
    summarized_context = summarize_pdf_content(context)
    questions = generate_questions_agent(summarized_context, max_questions)
    results = []
    for idx, question in enumerate(questions):
        st.write(f"Processing question {idx + 1}/{len(questions)}: {question}")
        result = process_question(question, summarized_context)
        results.append(result)
    return results

# ------------------------------
# Step 1: Upload & Q&A Generation (Storing in Qdrant with WordLlama Embeddings)
# ------------------------------
if selected_step == "Upload & Q&A Generation":
    st.title("Upload File and Generate Q&A Dataset")
    collection_name = st.text_input("Dataset Collection Name", value="")
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    num_questions = st.number_input("Number of Q&A (max. 50)", min_value=1, max_value=50, value=5, step=1)
    
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
        
        if st.button("Generate Q&A and Store in Qdrant"):
            with st.spinner("Generating Q&A pairs..."):
                qa_results = process_context(context_text, num_questions)
                
                # Create a CSV dataset (for download) with columns: prompt, completion, uuid.
                csv_buffer = StringIO()
                csv_writer = csv.writer(csv_buffer)
                csv_writer.writerow(["prompt", "completion", "uuid"])
                
                points = []
                for entry in qa_results:
                    doc_id = str(uuid.uuid4())
                    # Combine generic chain-of-thought and final answer in the completion field.
                    completion = entry['generic_cot'] + "\n\n" + entry['final_answer']
                    csv_writer.writerow([entry['question'], completion, doc_id])
                    
                    combined_text = entry['question'] + " " + entry['final_answer']
                    embedding_vector = wl.embed(combined_text)
                    if isinstance(embedding_vector, dict) and "embedding" in embedding_vector:
                        embedding_vector = embedding_vector["embedding"]
                    embedding_vector = flatten_embedding(embedding_vector)
                    if len(embedding_vector) != embedding_dimension:
                        st.error(f"Embedding dimension mismatch: expected {embedding_dimension}, got {len(embedding_vector)}")
                        continue
                    point = PointStruct(
                        id=doc_id,
                        vector=embedding_vector,
                        payload={
                            "question": entry['question'],
                            "generic_cot": entry['generic_cot'],
                            "final_answer": entry['final_answer']
                        }
                    )
                    points.append(point)
                
                if points:
                    qdrant_client.upsert(collection_name=qdrant_collection, points=points)
                    csv_data = csv_buffer.getvalue()
                    st.success("Dataset generated and stored in Qdrant successfully!")
                    st.download_button("Download CSV Dataset", data=csv_data, file_name="predibase_dataset.csv", mime="text/csv")
                    
                    json_data = json.dumps(qa_results, indent=4)
                    st.download_button("Download JSON Dataset", data=json_data, file_name="predibase_dataset.json", mime="application/json")
                else:
                    st.error("No valid embeddings were generated. Please check the debug output above.")

# ------------------------------
# Step 2: Fine-Tuning Job
# ------------------------------
if selected_step == "Fine-Tuning Job":
    st.title("Fine-Tune Model")
    dataset_name = st.text_input("Dataset Name", value="Enter Your dataset name")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"], key="ft_csv",
                                    help="This CSV file will be used for the fine-tuning job")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5, step=1)
    base_model_name = st.text_input("Base Model", value="llama-3-1-8b-instruct")
    repo_name = st.text_input("Repository Name", value="Enter adaptor name")
    repo_desc = st.text_input("Repository Description", value="Enter repo description")
    
    if st.button("Start Fine-Tuning") and uploaded_csv is not None:
        csv_file_path = "predibase_dataset_uploaded.csv"
        with open(csv_file_path, "wb") as f:
            f.write(uploaded_csv.read())
        dataset = pb.datasets.from_file(csv_file_path, name=dataset_name)
        repo = pb.repos.create(name=repo_name, description=repo_desc, exists_ok=True)
        adapter = pb.adapters.create(
            config=FinetuningConfig(
                base_model=base_model_name,
                epochs=epochs,
                rank=16,
                learning_rate=0.0002,
                target_modules=["q_proj", "v_proj", "k_proj"],
            ),
            dataset=dataset,
            repo=repo,
            description="Fine-tuning Q&A data"
        )
        st.success(f"Successfully requested fine-tuning of **{base_model_name}**.")

# ------------------------------
# Step 3: Query Embedding Database in Qdrant
# ------------------------------
if selected_step == "Query Embedding Database":
    st.title("Query Qdrant Embedding Database")
    query_text = st.text_input("Enter query")
    top_k = st.number_input("Number of results to return", min_value=1, max_value=20, value=5, step=1)
    if st.button("Search"):
        query_embedding = wl.embed(query_text)
        if isinstance(query_embedding, dict) and "embedding" in query_embedding:
            query_embedding = query_embedding["embedding"]
        query_embedding = flatten_embedding(query_embedding)
        search_result = qdrant_client.search(
            collection_name=qdrant_collection,
            query_vector=query_embedding,
            limit=top_k
        )
        if search_result:
            st.write("### Search Results:")
            for res in search_result:
                st.markdown(f"**ID:** {res.id}")
                st.markdown(f"**Question:** {res.payload.get('question', 'N/A')}")
                st.markdown(f"**Generic CoT:** {res.payload.get('generic_cot', 'N/A')}")
                st.markdown(f"**Final Answer:** {res.payload.get('final_answer', 'N/A')}")
                st.markdown("---")
        else:
            st.info("No matching results found.")

if selected_step == "Query Embedding Database" and st.button("Demo Similarity Ranking"):
    st.write("### WordLlama Similarity Ranking Demo")
    demo_query = "Machine learning methods"
    demo_candidates = [
        "Foundations of neural science",
        "Introduction to neural networks",
        "Cooking delicious pasta at home",
        "Introduction to philosophy: logic",
    ]
    sim_key = wl.key(demo_query)
    sorted_candidates = sorted(demo_candidates, key=sim_key, reverse=True)
    best_candidate = max(demo_candidates, key=sim_key)
    
    st.write("**Ranked Candidates:**")
    for i, candidate in enumerate(sorted_candidates, 1):
        st.write(f"{i}. {candidate} (Score: {sim_key(candidate):.4f})")
    st.write(f"\n**Best Match:** {best_candidate} (Score: {sim_key(best_candidate):.4f})")
