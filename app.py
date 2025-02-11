import streamlit as st
from transformers import pipeline

try:
    pipe = pipeline("text-generation", model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", trust_remote_code=True, device_map="auto") # device_map for potential GPU usage
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)

            with st.spinner("Processing your query, please wait..."):
                try:
                    messages = [{"role": "user", "content": user_input}]
                    response = pipe(messages, max_length=500, num_return_sequences=1)

                    if response:
                        st.write("Healthcare Assistant: ", response[0]['generated_text'])
                    else:
                        st.write("Healthcare Assistant: No response generated.")

                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.write("An error occurred. Please try again or rephrase your query.")

        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
    
