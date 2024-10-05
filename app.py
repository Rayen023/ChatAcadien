import streamlit as st
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# from langchain_cohere import CohereRerank
from langchain_voyageai import VoyageAIRerank
from langchain_voyageai import VoyageAIEmbeddings

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from datetime import datetime
import logging
import time
import re

from os import environ

logging.basicConfig(
    filename="logs.log",
    encoding="UTF-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


st.logo(
    "Images/avatarchat.png",  # Icon (displayed in sidebar)
    # link="https://streamlit.io/gallery",
    icon_image="Images/avatarchat.png",  # Alternate Icon if sidebar closed
    size="large",
)


# App title
st.set_page_config(
    page_title="ChatAcadien",
    page_icon="Images/avatarchat.png",
    # initial_sidebar_state="collapsed",
)


def get_env_variable(var_name):
    try:
        if var_name in environ:
            return environ[var_name]
        if var_name in st.secrets:
            return st.secrets[var_name]
    except Exception as e:
        logger.error("An error occurred while logging the conversation: %s", str(e))


def _get_session():
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(_get_session().id)


subject_to_email_french = {
    "Généalogie, Arbre de famille": "nadine.morin@umoncton.ca",
    "Numérisation, Bibliothèque, Livre, Don de livre": "nadine.morin@umoncton.ca",
    "Archives privées, Archives institutionnelles": "josee.theriault@umoncton.ca ou francois.j.leblanc@umoncton.ca",
    "Fonds, Don": "josee.theriault@umoncton.ca ou francois.j.leblanc@umoncton.ca",
    "Subvention": "francois.j.leblanc@umoncton.ca",
    "Folklore, Ethnologie, Conte, Légende, Musique, Tradition, Faits de folklore": "robert.richard@umoncton.ca",
    "Facebook, Médias sociaux, Événements": "erika.basque@umoncton.ca",
}
subject_to_email_english = {
    "Genealogy, Family tree": "nadine.morin@umoncton.ca",
    "Scan, Scanning, Library, Book, Donation of book": "nadine.morin@umoncton.ca",
    "Private archives, Institutional archives": "josee.theriault@umoncton.ca ou francois.j.leblanc@umoncton.ca",
    "Funds, Donation": "josee.theriault@umoncton.ca ou francois.j.leblanc@umoncton.ca",
    "Grant": "francois.j.leblanc@umoncton.ca",
    "Folklore, Ethnology, Tale, Legend, Music, Tradition, Folklore facts": "robert.richard@umoncton.ca",
    "Facebook, Social media, Events": "erika.basque@umoncton.ca",
}


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Comment puis-je vous aider ? | How can I help you ? ",
        }
    ]
    st.session_state["chat_id"] += "1"


english_strings = {
    "new_chat": "New chat",
    "contact": "For more information, Contact us :",
    "subject": "Please select the subject of your request.",
    "contact_options": "For the subject of {option}, Please contact : {subject_to_email}",
    "retrieval": "Retrieving relevant documents...",
    "writing": "Writing the response...",
    "completed": "Completed",
    "thinking": "Thinking...",
    "feedback": "Thanks for your feedback!",
    "submit_feedback": "Submit feedback",
    "feedback_placeholder": "Tell us more...",
}
french_strings = {
    "new_chat": "Nouveau chat ",
    "contact": "Pour plus d'informations, Contactez-nous :",
    "subject": "Veuillez sélectionner le sujet de votre demande.",
    "contact_options": "Pour le sujet de {option}, Veuillez contactez : {subject_to_email}",
    "retrieval": "Récupération des documents pertinents...",
    "writing": "Rédaction de la réponse...",
    "completed": "Terminé",
    "thinking": "Réflexion en cours...",
    "feedback": "Merci pour votre retour!",
    "submit_feedback": "Soumettre le retour",
    "feedback_placeholder": "Dites-nous en plus...",
}

if "language" not in st.session_state:
    st.session_state["language"] = "Français"

if st.session_state["language"] != "Français":
    subject_to_email = subject_to_email_english
    shown_strings = english_strings
else:
    shown_strings = french_strings
    subject_to_email = subject_to_email_french

subjects = sorted(subject_to_email.keys())


with st.sidebar:

    # st.title("Chat Acadien")

    st.button(
        shown_strings["new_chat"],
        on_click=clear_chat_history,
        icon=":material/edit_square:",
    )

    st.divider()
    language = st.selectbox(
        label="language",
        options=("Français", "Anglais"),
        key="language",
        label_visibility="collapsed",
    )

    @st.dialog(shown_strings["contact"], width="large")
    @st.fragment
    def contact():
        option = st.selectbox(
            shown_strings["subject"],
            subjects,
            placeholder=shown_strings["subject"],
            index=None,
            label_visibility="collapsed",
        )
        if option:
            st.write(
                shown_strings["contact_options"].format(
                    option=option, subject_to_email=subject_to_email[option]
                )
            )

    if st.button(
        shown_strings["contact"],
        use_container_width=True,
        icon=":material/forward_to_inbox:",
    ):
        contact()


prompt = st.chat_input("Message ChatAcadien...")


# @st.fragment
def rerun_last_question():
    st.session_state["messages"].pop(-1)


@st.fragment
def save_chat_logs():
    try:
        client = MongoClient(get_env_variable("MONGO_URI"), server_api=ServerApi("1"))
        db = client["chatdb"]
        collection = db["conversation_logs"]

        if "messages" in st.session_state:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_id = st.session_state["chat_id"]
            collection.update_one(
                {"_id": chat_id},
                {
                    "$set": {
                        "timestamp": timestamp,
                        "messages": st.session_state["messages"],
                    }
                },
                upsert=True,
            )

    except Exception as e:
        logger.error("An error occurred while logging the conversation: %s", str(e))


@st.fragment
def log_feedback():
    st.toast(shown_strings["feedback"], icon=":material/thumbs_up_down:")


@st.fragment
def save_to_db(feedback_msg):
    log_feedback()
    try:
        client = MongoClient(get_env_variable("MONGO_URI"), server_api=ServerApi("1"))
        db = client["chatdb"]
        collection = db["feedback_logs"]

        if "messages" in st.session_state:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            collection.insert_one(
                {
                    "timestamp": timestamp,
                    "messages": st.session_state["messages"][-2:],
                    "feedback_msg": str(feedback_msg) if feedback_msg else "",
                },
            )

    except Exception as e:
        logger.error("An error occurred while logging the conversation: %s", str(e))


@st.fragment
def n_feedback():
    instr = shown_strings["feedback_placeholder"]
    with st.form("feedback", clear_on_submit=True, border=False):
        feedback_msg = st.text_input(
            instr,
            placeholder=instr,
            label_visibility="collapsed",
        )

        if st.form_submit_button(
            shown_strings["submit_feedback"], use_container_width=True
        ):
            save_to_db(feedback_msg)


def feedback():
    st.session_state["container"] = st.container(border=False)
    container = st.empty()
    with container:

        cols_dimensions = [7, 7, 100]
        col0, col1, col2 = st.session_state["container"].columns(
            cols_dimensions, gap="medium"
        )
        col0.button(
            ":material/autorenew:",
            on_click=rerun_last_question,
            key="rerun_last_question",
        )
        with col1.popover(":material/thumb_down:"):
            n_feedback()
    container.empty()

    save_chat_logs()


voyageai_embeddings = VoyageAIEmbeddings(model="voyage-3")


def create_custom_retriever_tool(index_name, k, top_n, description, embeddings_model):
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    compressor = VoyageAIRerank(model="rerank-2", top_k=top_n)
    # compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    retriever_tool = create_retriever_tool(
        compression_retriever,
        index_name,
        description,
    )

    return retriever_tool


# Utilisation pour CEAAC
ceaac_retriever_tool = create_custom_retriever_tool(
    index_name="ceaac-general-info-index",
    k=10,
    top_n=3,
    description="Pour les questions relatives à la ceaac (tarifs, Horaires, consultation des archives et des livres), vous devez utiliser cet outil.",
    embeddings_model=voyageai_embeddings,
)


ceaac_faq_tool = create_custom_retriever_tool(
    index_name="ceaac-questions-frequemment-posees-index",
    k=12,
    top_n=3,
    description="Cet outil contient certaines FAQ (questions fréquemment posées) avec les réponses suggérées par le centre CEAAC. Utilise cet outil en parallèle avec les autres outils si besoin.",
    embeddings_model=voyageai_embeddings,
)


genealogie_retriever_tool = create_custom_retriever_tool(
    index_name="genealogie-acadienne-index-1",
    k=10,
    top_n=1,
    description="Pour les questions relatives à la généalogie et aux familles acadiennes, vous devez utiliser cet outil. Les informations étant sensibles, assurez-vous de vérifier l'exactitude des noms, sachant que différentes personnes peuvent avoir le même nom. Demandez, si nécessaire, la possibilité d'obtenir plus d'informations. Ne répondez pas sans justification.",
    embeddings_model=voyageai_embeddings,
)

# patrimoine_retriever_tool = create_custom_retriever_tool(
#     index_name="patrimoine-acadien-index",
#     k=10,
#     top_n=3,
#     description="Pour les questions relatives au patrimoine acadien, vous devez utiliser cet outil.",
#     embeddings_model=voyageai_embeddings,
# )

search = TavilySearchResults(max_results=2)

tools = [
    search,
    ceaac_faq_tool,
    # patrimoine_retriever_tool,
    ceaac_retriever_tool,
    genealogie_retriever_tool,
]

history = ChatMessageHistory()

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Comment puis-je vous aider ? | How can I help you ?",
        }
    ]


# model = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     streaming=True,
# )


model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=8096,
    timeout=None,
    max_retries=2,
    streaming=True,
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"Vous êtes un assistant virtuel du Centre d'études acadiennes Anselme-Chiasson (CEAAC). Répondez dans la même langue que l'utilisateur. Si l'utilisateur écrit en anglais, répondez en anglais. Si l'utilisateur écrit en français, répondez en français. Vous avez accès à des outils qui vous fournissent des informations spécifiques sur le centre. Si vous n'êtes pas en mesure de répondre à la demande de l'utilisateur, orientez-le selon le sujet vers l'adresse e-mail appropriée en vous référant à ce dictionnaire : {'; '.join(f'{key}: {value}' for key, value in subject_to_email.items())}. Utilisez l'outil TavilySearch pour les questions générales ou les événements en temps réel, ainsi que pour les questions liées au patrimoine acadien (telles que l'histoire de l'acadie, des recettes, la cuisine ou la musique acadienne).",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Comment puis-je vous aider? | How can I help you?",
        }
    ]


def escape_dollar_signs(text):
    return re.sub(r"(\d)\$", r"\1\\$", text)


for message in st.session_state.messages:
    if message["role"] == "assistant":
        # Check and modify the content if it has a number followed by a dollar sign
        modified_content = escape_dollar_signs(message["content"])

        with st.chat_message(message["role"], avatar="Images/avatarchat.png"):
            st.write(modified_content)
            history.add_user_message("AI : " + message["content"])

    elif message["role"] == "user":
        with st.chat_message(message["role"], avatar="Images/avataruser.png"):
            st.write(message["content"])
            history.add_user_message(message["content"])


memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history", chat_memory=history
)

agent = create_tool_calling_agent(model, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="Images/avataruser.png"):
        st.write(prompt)

debugging = False
if debugging:

    @st.fragment
    def generate_response():
        st_callback = StreamlitCallbackHandler(
            st.chat_message("assistant", avatar="Images/avatarchat.png"),
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            max_thought_containers=4,
        )
        response = agent_executor.invoke(
            {"input": st.session_state.messages[-1]["content"]},
            {
                "callbacks": [st_callback],
            },
        )
        try:
            modified_content = escape_dollar_signs(response["output"][0]["text"])
        except:
            modified_content = escape_dollar_signs(response["output"])
        st.write(modified_content)
        message = {"role": "assistant", "content": modified_content}
        st.session_state.messages.append(message)

else:
    # TODO add try excepts
    @st.fragment
    def generate_response():
        placeholder = st.empty()

        status = placeholder.status(shown_strings["thinking"], expanded=False)

        with status:
            st.write(shown_strings["retrieval"])
            st.write(shown_strings["writing"])
            response = agent_executor.invoke(
                {"input": st.session_state.messages[-1]["content"]},
            )
            status.update(
                label=shown_strings["completed"], state="complete", expanded=False
            )
        placeholder.empty()

        try:
            modified_content = escape_dollar_signs(response["output"][0]["text"])
        except:
            modified_content = escape_dollar_signs(response["output"])
        message = {"role": "assistant", "content": modified_content}
        st.session_state.messages.append(message)
        with st.chat_message("assistant", avatar="Images/avatarchat.png"):
            message_placeholder = st.empty()
            full_response = ""

        for chunk in modified_content.split(" "):
            full_response += chunk + " "
            time.sleep(0.01)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)


def retry_until_success(func, max_retries=None, delay=1):
    retries = 0
    while True:
        try:
            return func()
        except Exception as e:
            retries += 1
            if max_retries is not None and retries >= max_retries:
                break
            time.sleep(delay)


@st.fragment
def generate_response_and_layout_feedback():
    retry_until_success(generate_response, max_retries=2)


if st.session_state.messages[-1]["role"] != "assistant":
    generate_response_and_layout_feedback()

if (
    len(st.session_state.messages) > 1
    and st.session_state.messages[-1]["role"] == "assistant"
):
    feedback()
else:
    st.session_state["container"] = st.empty()


# TODO add fragment and put disclaimer at bottom and test if contact and this can work en parallele
if ("disclaimer" not in st.session_state) and (len(st.session_state["messages"]) == 1):
    st.session_state["disclaimer"] = True
    with st.empty():
        for seconds in range(15):
            st.warning(
                """‎ **Français :** Cette conversation sera enregistrée afin d'améliorer davantage les capacités de ChatAcadien. Vous pouvez cliquer sur 👎 pour fournir des commentaires sur la qualité des réponses. Note : ChatAcadien peut faire des erreurs. Vérifiez en ligne pour des informations importantes ou contactez-nous.\n\n"""
                + """‎ **Anglais :** This conversation will be recorded in order to further improve ChatAcadien's capabilities. You can click 👎 to provide feedback on the quality of the responses. Note: ChatAcadien may make mistakes. Check online for important information or contact us.""",
                icon="💡",
            )
            time.sleep(1)
        st.write("")
