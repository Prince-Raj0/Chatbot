import streamlit as st
import nltk
import os
import tempfile
import importlib  # For dynamic import

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

    if not HF_AUTH_TOKEN:
        st.error("HF_AUTH_TOKEN secret not found or incorrect. Please set it in Streamlit Cloud (no spaces!).")
        return None

    try:
        # Dynamic import of torch within the function
        torch = importlib.import_module("torch")

        from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

        model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Correct model name

        config = LlamaConfig.from_pretrained(model_name, token=HF_AUTH_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_AUTH_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16, token=HF_AUTH_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", torch_dtype=torch.bfloat16)

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
    st.title("Health Assistant Chatbot")  # Updated title

    user_input = st.text_area("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            with st.spinner("Generating response..."):
                response = chatbot(user_input)
                st.write("Health Assistant: ", response)  # Updated assistant name
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()

