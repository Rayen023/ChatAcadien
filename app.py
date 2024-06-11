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
from sqlalchemy.sql import text

st.logo(
    "Images/avatarchat.png",  # Icon (displayed in sidebar)
    # link="https://streamlit.io/gallery",
    icon_image="Images/avatarchat.png",  # Alternate Icon if sidebar closed
)


# App title
st.set_page_config(
    page_title="ChatAcadien",
    page_icon="Images/avatarchat.png",
    # initial_sidebar_state="collapsed",
)

# Define a dictionary to map subjects to emails
subject_to_email = {
    "Généalogie": "nadine.morin@umoncton.ca",
    "Genealogy": "nadine.morin@umoncton.ca",
    "Arbre de famille": "nadine.morin@umoncton.ca",
    "Family tree": "nadine.morin@umoncton.ca",
    "Numérisation": "nadine.morin@umoncton.ca",
    "Scan": "nadine.morin@umoncton.ca",
    "Scanning": "nadine.morin@umoncton.ca",
    "Bibliothèque": "nadine.morin@umoncton.ca",
    "Library": "nadine.morin@umoncton.ca",
    "Livre": "nadine.morin@umoncton.ca",
    "Book": "nadine.morin@umoncton.ca",
    "Don de livre": "nadine.morin@umoncton.ca",
    "Archives privées": "erika.basque@umoncton.ca",
    "Archives institutionnelles": "erika.basque@umoncton.ca",
    "Fonds": "erika.basque@umoncton.ca",
    "Don": "erika.basque@umoncton.ca",
    "Donation": "erika.basque@umoncton.ca",
    "Subvention": "francois.j.leblanc@umoncton.ca",
    "Folklore": "robert.richard@umoncton.ca",
    "Ethnologie": "robert.richard@umoncton.ca",
    "Conte": "robert.richard@umoncton.ca",
    "Légende": "robert.richard@umoncton.ca",
    "Musique": "robert.richard@umoncton.ca",
    "Tradition": "robert.richard@umoncton.ca",
    "Faits de folklore": "robert.richard@umoncton.ca",
    "Facebook": "erika.basque@umoncton.ca",
    "Médias sociaux": "erika.basque@umoncton.ca",
    "Événements": "erika.basque@umoncton.ca",
}


# Get the list of subjects from the dictionary keys
subjects = sorted(subject_to_email.keys())

# Create a multiselect widget


# Display the corresponding emails based on the selected options

with st.sidebar:

    # st.title("Chat Acadien")

    popover = st.popover(
        "Pour plus d'informations, Contactez-nous :", use_container_width=True
    )
    with popover:
        st.write(
            "Généalogie, Arbre de Famille, Numérisation, Bibliothèque, Livre, Don de Livre : nadine.morin@umoncton.ca"
        )
        st.write(
            "Archives Privées, Archives Institutionnelles, Fonds, Don : josee.theriault@umoncton.ca"
        )
        st.write("Subvention : francois.j.leblanc@umoncton.ca")
        st.write(
            "Folklore, Ethnologie, Conte, Légende, Musique, Tradition : robert.richard@umoncton.ca"
        )
        st.write("Facebook, Médias Sociaux, Événements : erika.basque@umoncton.ca")

    @st.experimental_dialog("Pour plus d'informations, Contactez-nous :", width="large")
    def contact():
        option = st.selectbox(
            "Choisir sujet de la demande",
            subjects,
            placeholder="Choisir sujet de la demande",
            index=None,
            label_visibility="collapsed",
        )
        if option:
            st.write(
                f"Pour le sujet de {option}, Veuillez contactez : {subject_to_email[option]}"
            )

    if st.button("Pour plus d'informations, Contactez-nous :"):
        contact()


prompt = st.chat_input("Message ChatAcadien...")

logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)


def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(_get_session().id)
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
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="Images/avatarchat.png"):
            st.write(message["content"])
            history.add_ai_message(message["content"])
    elif message["role"] == "user":
        with st.chat_message(message["role"]):
            st.write(message["content"])
            history.add_user_message(message["content"])


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
    st.session_state["chat_id"] += "1"


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


@st.experimental_fragment
def generate_response():
    st_callback = StreamlitCallbackHandler(
        st.chat_message("assistant", avatar="Images/avatarchat.png"),
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


def save_chat_logs():
    # Create the SQL connection to chats_db as specified in your secrets file.
    conn = st.connection("chats_db", type="sql")

    # Insert some data with conn.session.
    with conn.session as s:
        # Create chat_logs table if it doesn't exist
        s.execute(
            text(
                "CREATE TABLE IF NOT EXISTS chat_logs (id TEXT PRIMARY KEY, json_messages TEXT);"
            )
        )

        # Capture chat messages
        if "messages" in st.session_state:
            json_messages = json.dumps(st.session_state["messages"])
            chat_id = st.session_state["chat_id"]

            # Insert or update chat log in chat_logs table
            s.execute(
                text(
                    "INSERT OR REPLACE INTO chat_logs (id, json_messages) VALUES (:id, :json_messages);"
                ),
                params=dict(id=chat_id, json_messages=json_messages),
            )

        s.commit()


if len(st.session_state["messages"]) > 1:
    # We set the space between the icons thanks to a share of 100
    cols_dimensions = [1, 8, 19.4, 3, 3, 3, 1]
    col0, col1, col2, col3, col4, col5, col6 = st.columns(cols_dimensions)

    with col1:
        st.button("🗑️ Clear Chat", on_click=clear_chat_history)

    with col3:
        st.button("🔁", on_click=rerun_last_question)

    with col4:
        st.button(p_icon, on_click=lambda: log_feedback(p_icon))

    with col5:
        st.button(n_icon, on_click=lambda: log_feedback(n_icon))

    save_chat_logs()


import time

# TODO add fragment and put disclaimer at bottom and test if contact and this can work en parallele
if ("disclaimer" not in st.session_state) and (len(st.session_state["messages"]) == 1):
    st.session_state["disclaimer"] = True
    with st.empty():
        for seconds in range(10):
            st.warning(
                """‎ Cette conversation sera enregistrée afin d'améliorer davantage les capacités de ChatAcadien. Vous pouvez cliquer sur 👍 ou 👎 pour fournir des commentaires sur la qualité des réponses. Note : ChatAcadien peut faire des erreurs. Vérifiez en ligne pour des informations importantes ou contactez-nous.""",
                icon="💡",
            )
            time.sleep(1)
        st.write("")
