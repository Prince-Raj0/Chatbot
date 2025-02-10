import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForCausalLM, pipeline
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

# 2. Download NLTK data (using the correct resource name and path)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH)

# 3. Load Mistral model (with enhanced error handling and caching, using st.secrets)
@st.cache_resource
def load_mistral_model():
    HF_AUTH_TOKEN = st.secrets.get("HF_AUTH_TOKEN")  # Get token from secrets

    if not HF_AUTH_TOKEN:  # Check if the token is available
        st.error("HF_AUTH_TOKEN secret not found. Please set it in Streamlit Cloud.")
        return None  # Return None if the token is not available

    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Small-24B-Instruct-2501", token=HF_AUTH_TOKEN)  # Use token here
        model = TFAutoModelForCausalLM.from_pretrained("mistralai/Mistral-Small-24B-Instruct-2501", token=HF_AUTH_TOKEN)  # Use token here
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)  # No device_map or torch_dtype here for TensorFlow
        return pipe
    except Exception as e:
        st.error(f"Error loading Mistral model: {e}")
        return None

mistral_pipe = load_mistral_model()

# Define healthcare-specific response logic (using Mistral)
def healthcare_chatbot(user_input):
    if mistral_pipe is None:
        return "Sorry, the model is not loaded. Please try again later."

    try:
        messages = [{"role": "user", "content": user_input}]  # Format as messages for Mistral
        response = mistral_pipe(messages, max_new_tokens=300)

        if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
            return response[0]['generated_text']
        else:
            return "I'm still learning. Please try another query."  # Handle unexpected response format

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I encountered an error. Please try again."


# Streamlit web app interface
def main():
    st.title("Healthcare Assistant Chatbot")

    user_input = st.text_area("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a query.")


if __name__ == "__main__":
    main()
      
