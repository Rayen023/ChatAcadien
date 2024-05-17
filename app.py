import streamlit as st

# App title
st.set_page_config(page_title="ChatAcadien", page_icon="💬")
# st.set_page_config(page_title="My Streamlit App", page_icon=":moon:", layout="wide", initial_sidebar_state="auto", )


with st.sidebar:

    ms = st.session_state
    if "themes" not in ms:
        ms.themes = {
            "current_theme": "light",
            "refreshed": True,
            "light": {
                "theme.base": "dark",
                # "theme.backgroundColor": "black",
                # "theme.primaryColor": "#c98bdb",
                # "theme.secondaryBackgroundColor": "#5591f5",
                "theme.textColor": "white",
                "theme.textColor": "white",
                "button_face": "🌘",
            },
            "dark": {
                "theme.base": "light",
                # "theme.backgroundColor": "white",
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
    # st.subheader("Models and parameters")

    # temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=5.0, value=0.1, step=0.01)
    # top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    # max_length = st.sidebar.slider("max_length", min_value=64, max_value=4096, value=512, step=8)

prompt = st.chat_input("Message ChatAcadien...")

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


## Statefully manage chat history ###
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug

set_debug(True)

store = {}
session_id = "default"


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = StreamlitChatMessageHistory(key="chat_messages")
    return store[session_id]


llama_api_key = (
    "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"
)


class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(
        self,
        model_name: str,
        openai_api_key: str = llama_api_key,
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


llm = ChatOpenRouter(
    model_name="meta-llama/llama-3-8b-instruct:free",
    temperature=0.0,
)

qa_system_prompt = """You are an AI chatbot having a conversation with a human .\
"""
general_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

general_chain = general_prompt | llm | StrOutputParser()

conv_general_chain = RunnableWithMessageHistory(
    general_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    # output_messages_key="answer",
)


# @st.cache_resource(ttl=600, show_spinner=False)
def conversational_chat(query, session_id):
    response = conv_general_chain.stream(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    return response


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
    store[session_id].clear()


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# @st.cache_resource()
def generate_response(prompt_input):
    if not prompt_input:
        return
    output = conversational_chat(str(prompt_input), session_id=session_id)
    return output


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        if response:
            msg_ = st.write_stream(response)
    message = {"role": "assistant", "content": msg_}
    st.session_state.messages.append(message)
