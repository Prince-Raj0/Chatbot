import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForCausalLM, pipeline  # TensorFlow version
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

# 3. Load Mistral model (using st.secrets)
@st.cache_resource
def load_mistral_model():
    HF_AUTH_TOKEN = st.secrets.get("HF_AUTH_TOKEN")

    # Debugging: Print token and length (remove after testing)
    st.write(f"HF_AUTH_TOKEN (len: {len(HF_AUTH_TOKEN) if HF_AUTH_TOKEN else 0}): '{HF_AUTH_TOKEN}'")

    if not HF_AUTH_TOKEN:
        st.error("HF_AUTH_TOKEN secret not found or incorrect. Please set it in Streamlit Cloud (no spaces!).")
        return None

    try:
        # Import torch ONLY inside this function (for transformers dependency):
        import torch  # This is crucial, even with TensorFlow

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Small-24B-Instruct-2501", token=HF_AUTH_TOKEN)
        model = TFAutoModelForCausalLM.from_pretrained("mistralai/Mistral-Small-24B-Instruct-2501", token=HF_AUTH_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)  # No device_map or torch_dtype for TensorFlow
        return pipe
    except Exception as e:
        st.error(f"Error loading Mistral model: {e}")
        return None

mistral_pipe = load_mistral_model()

# 4. Healthcare Chatbot Logic
def healthcare_chatbot(user_input):
    if mistral_pipe is None:
        return "Sorry, the model is not loaded. Please try again later."

    try:
        messages = [{"role": "user", "content": user_input}]
        response = mistral_pipe(messages, max_new_tokens=300)

        if isinstance(response, list) and len(response) > 0 and 'generated_text' in response:
            generated_text = response['generated_text']
            return generated_text
        elif isinstance(response, list) and len(response) > 0 and isinstance(response, dict) and 'generated_text' in response:
            generated_text = response['generated_text']
            return generated_text
        elif isinstance(response, str):
            return response
        else:
            return "I'm still learning. Please try another query."

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I encountered an error. Please try again."


# 5. Streamlit Web App
def main():
    st.title("Healthcare Assistant Chatbot")

    user_input = st.text_area("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            with st.spinner("Generating response..."):
                response = healthcare_chatbot(user_input)
                st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a query.")


if __name__ == "__main__":
    main()
                                              
