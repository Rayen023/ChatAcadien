import os
from datetime import datetime
import logging
import json

import streamlit as st
import tiktoken

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_pinecone import PineconeVectorStore


st.logo(
    "Images/chat_logo.png",  # Icon (displayed in sidebar)
    # link="https://streamlit.io/gallery",
    icon_image="Images/chat_logo.png",  # Alternate Icon if sidebar closed
)


# App title
st.set_page_config(page_title="ChatAcadien", page_icon="Images/chat_logo.png")
# st.set_page_config(page_title="My Streamlit App", page_icon=":moon:", layout="wide", initial_sidebar_state="auto", )

with st.sidebar:

    st.title("Chat Acadien")

prompt = st.chat_input("Message ChatAcadien...")

logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)


p_icon = "👍"
n_icon = "👎"


def log_feedback(icon):
    # We display a nice toast
    st.toast("Thanks for your feedback!", icon=":material/thumbs_up_down:")

    # We retrieve the last question and answer
    last_messages = json.dumps(st.session_state["messages"][-2:])

    # We record the timestamp
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "

    # And include the messages
    activity += "positive" if icon == p_icon else "negative"
    activity += ": " + last_messages

    # And log everything
    logger.info(activity)


embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
)

index_name = "docs-quickstart-index"

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7},
)
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

compressor = CohereRerank(
    model="rerank-multilingual-v3.0",
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


retriever_tool = create_retriever_tool(
    compression_retriever,
    "computer_vision_and_defect_detection_search",
    "If the question is related to computer vision or defect detection, you must use this tool. When using this tool, for the query key, pass an initial detailed paragraph answer as to enhance this tool's retrieval search. ",
)

search = TavilySearchResults(max_results=2)

tools = [search, retriever_tool]

history = ChatMessageHistory()

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Comment puisse-je vous aidez?"}
    ]

functions = [convert_to_openai_function(f) for f in tools]
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    streaming=True,
).bind(functions=functions)


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Comment puisse-je vous aidez?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])


memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history", chat_memory=history
)

agent_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    )
    | prompt_template
    | model
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent_chain, tools=tools, verbose=True, memory=memory
)


@st.experimental_fragment
def rerun_last_question():
    st.session_state["messages"].pop(-1)


@st.experimental_fragment
def clear_chat_history():  # TODO
    st.session_state.messages = [
        {"role": "assistant", "content": "Comment puisse-je vous aidez?"}
    ]


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


@st.experimental_fragment
def generate_response():
    st_callback = StreamlitCallbackHandler(
        st.chat_message("assistant"),
        expand_new_thoughts=True,
        collapse_completed_thoughts=False,
        max_thought_containers=0,
    )
    response = agent_executor.invoke(
        {"input": st.session_state.messages[-1]["content"]},
        {"callbacks": [st_callback]},
    )
    message = {"role": "assistant", "content": response["output"]}
    st.session_state.messages.append(message)


if st.session_state.messages[-1]["role"] != "assistant":
    generate_response()


if len(st.session_state["messages"]) > 1:

    # We set the space between the icons thanks to a share of 100
    cols_dimensions = [7, 19.4, 19.3, 9, 8.6, 8.6, 28.1]
    col0, col1, col2, col3, col4, col5, col6 = st.columns(cols_dimensions)

    with col1:

        # Converts the list of messages into a JSON Bytes format
        json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")

        # And the corresponding Download button
        st.download_button(
            label="📥 Save chat",
            data=json_messages,
            file_name="chat_conversation.json",
            mime="application/json",
        )

    with col2:
        st.button("🗑️ Clear Chat", on_click=clear_chat_history)

    with col3:
        st.button("🔁", on_click=rerun_last_question)

    with col4:
        st.button(p_icon, on_click=lambda: log_feedback(p_icon))

    with col5:
        st.button(n_icon, on_click=lambda: log_feedback(n_icon))

    with col6:

        # We initiate a tokenizer
        enc = tiktoken.get_encoding("cl100k_base")

        # We encode the messages
        tokenized_full_text = enc.encode(
            " ".join([item["content"] for item in st.session_state["messages"]])
        )

        # And display the corresponding number of tokens
        label = f"💬 {len(tokenized_full_text)} tokens"
        st.button(label)

# else:

#     # At the first run of a session, we temporarly display a message
#     if "disclaimer" not in st.session_state:
#         with st.empty():
#             for seconds in range(3):
#                 st.warning(
#                     "‎ You can click on 👍 or 👎 to provide feedback regarding the quality of responses.",
#                     icon="💡",
#                 )
#                 time.sleep(1)
#             st.write("")
#             st.session_state["disclaimer"] = True
