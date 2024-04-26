import streamlit as st
import requests
import json

OPENROUTER_API_KEY = (
    "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"
)
gemma_api_key = (
    "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"
)

# App title
st.set_page_config(page_title="🦙💬 ChatAcadien")
# st.set_page_config(page_title="My Streamlit App", page_icon=":moon:", layout="wide", initial_sidebar_state="auto", )


with st.sidebar:

    ms = st.session_state
    if "themes" not in ms:
        ms.themes = {
            "current_theme": "light",
            "refreshed": True,
            "light": {
                "theme.base": "dark",
                "theme.backgroundColor": "black",
                # "theme.primaryColor": "#c98bdb",
                # "theme.secondaryBackgroundColor": "#5591f5",
                "theme.textColor": "white",
                "theme.textColor": "white",
                "button_face": "🌘",
            },
            "dark": {
                "theme.base": "light",
                "theme.backgroundColor": "white",
                # "theme.primaryColor": "#5591f5",
                # "theme.secondaryBackgroundColor": "#82E1D7",
                "theme.textColor": "#0a1464",
                "button_face": "🌖",
            },
        }

    def ChangeTheme():
        previous_theme = ms.themes["current_theme"]
        tdict = (
            ms.themes["light"]
            if ms.themes["current_theme"] == "light"
            else ms.themes["dark"]
        )
        for vkey, vval in tdict.items():
            if vkey.startswith("theme"):
                st._config.set_option(vkey, vval)

        ms.themes["refreshed"] = False
        if previous_theme == "dark":
            ms.themes["current_theme"] = "light"
        elif previous_theme == "light":
            ms.themes["current_theme"] = "dark"

    btn_face = (
        ms.themes["light"]["button_face"]
        if ms.themes["current_theme"] == "light"
        else ms.themes["dark"]["button_face"]
    )
    st.button(btn_face, on_click=ChangeTheme)

    if ms.themes["refreshed"] == False:
        ms.themes["refreshed"] = True
        st.rerun()

    st.title("ChatAcadien")
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

prompt = st.chat_input("Message ChatAcadien...")

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

msgs = StreamlitChatMessageHistory(key="chat_messages")


class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(
        self,
        model_name: str,
        openai_api_key: str = gemma_api_key,
        openai_api_base: str = "https://openrouter.ai/api/v1",
        **kwargs,
    ):
        openai_api_key = openai_api_key
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs,
        )


@st.cache_resource
def conversational_chat(query):
    llm = ChatOpenRouter(
        model_name="google/gemma-7b-it:free",
        temperature=temperature,
        # max_tokens=max_length,
        top_p=top_p,
    )
    memory = ConversationBufferMemory(
        memory_key="history", chat_memory=msgs, return_messages=True
    )
    chain = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI chatbot having a conversation with a human."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        ),
        memory=memory,
        verbose=True,
    )

    result = chain({"input": query})
    st.session_state["history"].append((query, result["text"]))
    return result["text"]


if "history" not in st.session_state:
    st.session_state["history"] = []


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Comment puisse-je vous aidez?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():  # TODO
    st.session_state.messages = [
        {"role": "assistant", "content": "Comment puisse-je vous aidez?"}
    ]
    msgs.clear()
    st.session_state["history"] = []


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# @st.cache_resource()
def generate_response(prompt_input):
    if not prompt_input:
        return
    output = conversational_chat(str(prompt_input))
    return output


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ""
            if response:
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
