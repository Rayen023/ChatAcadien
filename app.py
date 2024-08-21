import streamlit as st
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_cohere import CohereRerank, CohereEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.retriever import create_retriever_tool

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from datetime import datetime
import logging
import json
import time
import re


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
)


# App title
st.set_page_config(
    page_title="ChatAcadien",
    page_icon="Images/avatarchat.png",
    # initial_sidebar_state="collapsed",
)


def _get_session():
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(_get_session().id)


subject_to_email = {
    "Généalogie, Genealogy, Arbre de famille, Family tree": "nadine.morin@umoncton.ca",
    "Numérisation, Scan, Scanning, Bibliothèque, Library, Livre, Book, Don de livre": "nadine.morin@umoncton.ca",
    "Archives privées, Archives institutionnelles": "josee.theriault@umoncton.ca ou francois.j.leblanc@umoncton.ca",
    "Fonds, Don, Donation": "josee.theriault@umoncton.ca ou francois.j.leblanc@umoncton.ca",
    "Subvention": "francois.j.leblanc@umoncton.ca",
    "Folklore, Ethnologie, Conte, Légende, Musique, Tradition, Faits de folklore": "robert.richard@umoncton.ca",
    "Facebook, Médias sociaux, Événements": "erika.basque@umoncton.ca",
}


subjects = sorted(subject_to_email.keys())


@st.fragment
def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Comment puis-je vous aider? | How can I help you? ",
        }
    ]
    st.session_state["chat_id"] += "1"


with st.sidebar:

    # st.title("Chat Acadien")

    st.button(":pencil2: New chat", on_click=clear_chat_history)

    st.divider()

    @st.dialog("Pour plus d'informations, Contactez-nous :", width="large")
    @st.fragment
    def contact():
        option = st.selectbox(
            "Veuillez sélectionner le sujet de votre demande.",
            subjects,
            placeholder="Veuillez sélectionner le sujet de votre demande.",
            index=None,
            label_visibility="collapsed",
        )
        if option:
            st.write(
                f"Pour le sujet de {option}, Veuillez contactez : {subject_to_email[option]}"
            )

    if st.button("Pour plus d'informations, Contactez-nous :"):
        contact()

    popover2 = st.popover(
        "Pour plus d'informations, Contactez-nous :", use_container_width=True
    )
    with popover2:
        option2 = popover2.selectbox(
            "Choisir sujet de la demande",
            subjects,
            placeholder="Choisir sujet ...",
            index=None,
            label_visibility="collapsed",
            key="option2",
        )
        if option2:
            popover2.write(
                f"Pour le sujet de {option2}, Veuillez contactez : {subject_to_email[option2]}"
            )


prompt = st.chat_input("Message ChatAcadien...")


@st.fragment
def rerun_last_question():
    st.session_state["messages"].pop(-1)


@st.fragment
def save_chat_logs():
    try:
        client = MongoClient(st.secrets["mongo"]["uri"], server_api=ServerApi("1"))
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
    st.toast("Thanks for your feedback!", icon=":material/thumbs_up_down:")


@st.fragment
def save_to_db(feedback_msg):
    log_feedback()
    try:
        client = MongoClient(st.secrets["mongo"]["uri"], server_api=ServerApi("1"))
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
    instr = "Tell us more.. "
    with st.form("feedback", clear_on_submit=True, border=False):
        feedback_msg = st.text_input(
            instr,
            placeholder=instr,
            label_visibility="collapsed",
        )

        if st.form_submit_button("Submit feedback", use_container_width=True):
            save_to_db(feedback_msg)
            rerun_last_question()
            st.rerun()


def feedback():
    # feedback = streamlit_feedback(
    #     feedback_type="thumbs",
    #     align="flex-end",
    #     # optional_text_label="[Optional] Please provide an explanation",
    #     on_submit=on_submit,
    #     key="feedback",
    # )
    # copy lil chat messages w si feedback nappendi lfeedback w titsava l copy lmongo
    # feedback
    container = st.container(border=False)

    cols_dimensions = [85, 7, 7, 3]
    col0, col1, col2, col3 = container.columns(cols_dimensions)
    col1.button("🔁", on_click=rerun_last_question, key="rerun_last_question")
    with col2.popover("👎"):
        n_feedback()

    save_chat_logs()


embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
)


def create_custom_retriever_tool(index_name, k, top_n, description):
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)

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
)

# Utilisation pour la généalogie
genealogie_retriever_tool = create_custom_retriever_tool(
    index_name="arbre-de-familles-acadiennes-index",
    k=100,
    top_n=4,
    # description="Pour les questions relatives à la généalogie et à l'arbre des familles acadiennes, vous devez utiliser cet outil. Ces informations sont tres sensibles et il ne faut retourner des informations erronees donc verifie l'existence des noms exactes et si tu les trouves pas alors retourne que tu es pas sur et demande les de contacter le ceaac. Si plusieurs personnes aient des noms similaires comme (Charles Melanson et Anne Broussard) ou (Charles Melanson et Anne Léger), veuillez retounrer les differents familles comme options. Verifie aussi les familles qui n'ont pas eu d'enfants comme Charles Melanson et Anne Broussard",
    # description="Pour les questions relatives à la généalogie et à l'arbre des familles acadiennes, vous devez utiliser cet outil. Mais ne reformulez pas une réponse.!! Vous devez lire les données et en extraire les informations que vous trouvez pertinentes. Vous devez extraire ces paragraphes sans reformulation !! tels qu'ils sont et les retourner à l'utilisateur. Tu dois retourner l'extrait et les parties de textes qui l'entourent (toute les section relatives depuis un numero alphabetique). Cela permet à l'utilisateur de faire lui-même les liaisons pour que l'information soit plus précise et authentique. !. POur que tu comprenne mieux les donnees il faut savvoir les parents se trouvent entre parenthese exemple Marguerite ROBICHAUD (Dominique & Geneviève) ou Jean LÉGER (Joseph & Anne Gaudet). et que les enfants sont sous Enfants : avec les numeros romains non alphabetique",
    description="""Pour les questions concernant la généalogie et l'arbre des familles acadiennes, utilisez cet outil.
Pour chaque nombre arabe, il s’agit d’un nouveau groupe familial. Les parents sont indiqués entre parenthèses, par exemple, Marguerite ROBICHAUD (Dominique & Geneviève), ce qui signifie que les parents de Marguerite ROBICHAUD sont Dominique et Geneviève. Les enfants sont listés sous "Enfants :" avec des numéros romains.
Lisez les données et retournez des sections extraits sans les reformuler et sans modifications de ta part. Retourne tout le groupe familial lies a la question depuis son debut pas seulement une ligne, n'essaye pas de faire de conclusion toi. Laisse l'utilisateur faire les connexions nécessaires pour une information plus précise et authentique.
Il faut savoir que les utilisateurs peuvent donner des informations ou des liaisons incorrectes, si l'information n'est pas citee clairement dans le texte, demande de contacter le centre.
""",
)

search = TavilySearchResults(max_results=2)

tools = [search, ceaac_retriever_tool, genealogie_retriever_tool]

history = ChatMessageHistory()

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Comment puis-je vous aider? | How can I help you? ",
        }
    ]

functions = [convert_to_openai_function(f) for f in tools]
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True,
).bind(functions=functions)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"Vous êtes un assistant virtuel du Centre d'études acadiennes Anselme-Chiasson (CEAAC). Répondez dans la même langue que l'utilisateur. Si l'utilisateur écrit en anglais, répondez en anglais. Si l'utilisateur écrit en français, répondez en français. Vous avez accès à des outils qui vous fournissent des informations spécifiques sur le centre. Si vous n'êtes pas en mesure de répondre à la demande de l'utilisateur, orientez-le selon le sujet vers l'adresse e-mail appropriée en vous référant à ce dictionnaire {'; '.join(f'{key}: {value}' for key, value in subject_to_email.items())}.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
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
    agent=agent_chain,
    tools=tools,
    verbose=True,
    memory=memory,
)


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


@st.fragment
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
    modified_content = escape_dollar_signs(response["output"])
    message = {"role": "assistant", "content": modified_content}
    st.session_state.messages.append(message)


@st.fragment
def test_fn():
    generate_response()
    placeholder = st.empty()
    if len(st.session_state.messages) > 1:
        with placeholder:
            feedback()


if st.session_state.messages[-1]["role"] != "assistant":
    test_fn()


# TODO add fragment and put disclaimer at bottom and test if contact and this can work en parallele
if ("disclaimer" not in st.session_state) and (len(st.session_state["messages"]) == 1):
    st.session_state["disclaimer"] = True
    with st.empty():
        for seconds in range(15):
            st.warning(
                """‎ Cette conversation sera enregistrée afin d'améliorer davantage les capacités de ChatAcadien. Vous pouvez cliquer sur 👎 pour fournir des commentaires sur la qualité des réponses. Note : ChatAcadien peut faire des erreurs. Vérifiez en ligne pour des informations importantes ou contactez-nous.""",
                icon="💡",
            )
            time.sleep(1)
        st.write("")
