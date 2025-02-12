import streamlit as st
import nltk
import os
import tempfile
import tensorflow as tf

# 1. NLTK Data Path (using tempfile)
try:
    NLTK_DATA_PATH = os.path.join(tempfile.gettempdir(), "nltk_data")
except:
    NLTK_DATA_PATH = ".nltk_data"

if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)

nltk.data.path.append(NLTK_DATA_PATH)

# 2. Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH)

@st.cache_resource
def load_llama_model():
    HF_AUTH_TOKEN = st.secrets.get("HF_AUTH_TOKEN")

    st.write(f"HF_AUTH_TOKEN: '{HF_AUTH_TOKEN}' (len: {len(HF_AUTH_TOKEN) if HF_AUTH_TOKEN else 0})")  # Debugging

    if not HF_AUTH_TOKEN:
        st.error("HF_AUTH_TOKEN secret not found or incorrect. Please set it in Streamlit Cloud (no spaces!).")
        return None

    try:
        from transformers import pipeline, AutoTokenizer, TFAutoModelForCausalLM

        # Updated Model name
        model_name = "meta-llama/Llama-3.1-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_AUTH_TOKEN)
        model = TFAutoModelForCausalLM.from_pretrained(model_name, token=HF_AUTH_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        return pipe
    except Exception as e:
        st.error(f"Error loading Llama model: {e}")
        return None

llama_pipe = load_llama_model()

def chatbot(user_input):
    if llama_pipe is None:
        return "Sorry, the model is not loaded. Please try again later."

    try:
        messages = [{"role": "user", "content": user_input}]
        response = llama_pipe(messages, max_new_tokens=300)

        if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
            generated_text = response[0]['generated_text']
            return generated_text
        elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) and 'generated_text' in response[0]:
            generated_text = response[0]['generated_text']
            return generated_text
        elif isinstance(response, str):
            return response
        else:
            return "I'm still learning. Please try another query."

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I encountered an error. Please try again."

def main():
    st.title("Llama 3.1 Chatbot")  # Updated title

    user_input = st.text_area("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            with st.spinner("Generating response..."):
                response = chatbot(user_input)
                st.write("Llama 3.1 Assistant: ", response)  # Updated assistant name
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()

