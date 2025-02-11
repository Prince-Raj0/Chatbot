import streamlit as st
from transformers import pipeline

# Load the DeepSeek model pipeline (handle potential errors)
try:
    pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if model loading fails

# Streamlit web app interface
def main():
    st.title("Healthcare Assistant Chatbot (Beta - for informational purposes only)")
    st.markdown("**Disclaimer:** This chatbot is for informational purposes only and should not be considered medical advice. Always consult with a qualified healthcare professional for any health concerns.")

    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)

            with st.spinner("Processing your query, please wait..."):
                try:
                    messages = [{"role": "user", "content": user_input}]
                    response = pipe(messages, max_length=500, num_return_sequences=1)

                    if response: # Check if response is not empty
                        st.write("Healthcare Assistant: ", response[0]['generated_text'])
                    else:
                        st.write("Healthcare Assistant: No response generated.")

                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.write("An error occurred. Please try again or rephrase your query.")  # More user-friendly message

        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
            
