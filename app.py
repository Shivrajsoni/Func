import streamlit as st
import requests
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nomic import embed

# Load pre-embedded data
with open("pu_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]
embeddings = np.array(data["embeddings"])

# Semantic Search using Nomic
def retrieve_context(query, top_k=3):
    try:
        query_vec = embed.text([query])[0]  # no np.array() here
        query_vec = np.array(query_vec).reshape(1, -1)
        scores = cosine_similarity(query_vec, embeddings)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [texts[i] for i in top_indices]
    except Exception as e:
        return [f"❌ Error retrieving context: {e}"]

# Ask LLaMA running on Ollama
def ask_llama(query, history):
    context_chunks = retrieve_context(query)
    context = "\n\n".join(context_chunks)
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])

    prompt = f"""You are a helpful assistant for Panjab University admissions.
Use the context and past conversation to answer the question.

Context:
{context}

Conversation so far:
{history_text}

Now answer this:
User: {query}
Assistant:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "❌ No response returned").strip()
    except Exception as e:
        return f"❌ Error contacting LLaMA: {e}"

# Streamlit UI
st.title("🎓 PU Admissions Chatbot (Nomic + LLaMA3)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask your admission-related question...")

if user_input:
    with st.spinner("Thinking..."):
        answer = ask_llama(user_input, st.session_state.history)
        st.session_state.history.append((user_input, answer))

# Show conversation
for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)
