import streamlit as st
import requests

# App title
st.set_page_config(page_title="🦙💬 AcadienGPT")

with st.sidebar:
    st.title("AcadienGPT Chatbot")
    st.subheader("Models and parameters")

    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=5.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider(
        "top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    )
    max_length = st.sidebar.slider(
        "max_length", min_value=64, max_value=4096, value=512, step=8
    )

prompt = st.chat_input("Message AcadienGPT...")


API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
)
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = query(
        {
            "inputs": prompt,
        }
    )
    return output[0]["generated_text"]


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
