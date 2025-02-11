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

# ------------------------------
# Qdrant Remote Client (Cloud Qdrant)
# ------------------------------
QDRANT_URL = "https://9c8527a0-e8d7-4807-950a-a705b338b515.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzcwNzg5NzQ0fQ.7XgHym0bM4-t0qdXqxYqbYkcgUCElhFla9ITsTdT86A"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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

# ------------------------------
# Helper Functions: Querying Qdrant
# ------------------------------
def query_qdrant(query_text: str, top_k: int = 5):
    """Queries Qdrant for similar embeddings."""
    query_embedding = wl.embed(query_text)
    if isinstance(query_embedding, dict) and "embedding" in query_embedding:
        query_embedding = query_embedding["embedding"]
    query_embedding = flatten_embedding(query_embedding)

    search_result = qdrant_client.search(
        collection_name=qdrant_collection,
        query_vector=query_embedding,
        limit=top_k
    )

    return search_result

# ------------------------------
# Streamlit Interface
# ------------------------------
st.title("Qdrant Cloud Q&A System")

query_text = st.text_input("Enter your query:")
top_k = st.number_input("Number of results to return", min_value=1, max_value=20, value=5, step=1)

if st.button("Search"):
    results = query_qdrant(query_text, top_k)
    if results:
        st.write("### Search Results:")
        for res in results:
            st.markdown(f"**ID:** {res.id}")
            st.markdown(f"**Question:** {res.payload.get('question', 'N/A')}")
            st.markdown(f"**Generic CoT:** {res.payload.get('generic_cot', 'N/A')}")
            st.markdown(f"**Final Answer:** {res.payload.get('final_answer', 'N/A')}")
            st.markdown("---")
    else:
        st.info("No matching results found.")

