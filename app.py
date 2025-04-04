import asyncio
import logging
import re
import time
from datetime import datetime
from os import environ

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.tools.retriever import create_retriever_tool
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import BraveSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

# from langchain_core.tools import tool
# from langchain.retrievers.document_compressors import LLMListwiseRerank


DEBUGGING = False

logging.basicConfig(
    filename="logs.log",
    encoding="UTF-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


st.logo(
    "Images/logo2.png",  # Icon (displayed in sidebar)
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

# Set default model in Streamlit session state
if "DEFAULT_MODEL_NAME" not in st.session_state:
    st.session_state["DEFAULT_MODEL_NAME"] = (
        "claude-3-5-sonnet-latest"
        # "openai/gpt-4o"
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


def show_chatacadien_docs():
    with st.expander("Documents ChatAcadien", expanded=False):
        # First section: CEAAC Documents
        st.markdown("### Documents Centre d'études acadiennes Anselme-Chiasson (CEAAC)")
        st.markdown(
            """
        - Politique accès 2020
        - Infos-Site-Web
        - Grille tarifaire
        """
        )

        # Second section: Genealogical Documents
        st.markdown("### Registres généalogiques (1700-1900)")

        # Join family names with commas for a more compact display
        families = "Allain, Arsenault, Babin, Babineau, Bastarache, Belliveau, Bordage, Boucher, Boudreau, Bourgeois, Bourque, Caissie, Collette, Cormier, Daigle, Devarennes, Doiron, Gaudet, Gautreau, Girouard, Goguen, Gosselin, Hache, Landry, Leblanc, Leger, Maillet, Martin, Melanson, Petitpas, Poirier, Richard, Robichaud, Savoie, Surette, Thibodeau, Vautour"

        st.markdown("**Noms de familles :** " + families)
        st.markdown(
            "*Source : La généalogie des trente-sept familles hôtesses des « Retrouvailles 94 »*"
        )


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

sidebar_style = """
<style>
.sidebar-main {
    padding-bottom: 18vmax ; /* Adjust this value based on your footer height */
}
.st-emotion-cache-1gwvy71 {
    padding-bottom: 0vh !important;
}
</style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)

with st.sidebar:

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

    st.selectbox(
        " ",
        range(2024, 1900, -1),
        index=None,
        key="years_limit",
        label_visibility="collapsed",
        placeholder="Filtrer recherche web depuis :",
    )

    show_chatacadien_docs()

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

    st.markdown(
        "<h6 style='text-align: center; color: gray; font-size: 11px;'>ChatAcadien peut faire des erreurs. Verifiez les informations importantes.</h6>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-main"> </div>', unsafe_allow_html=True)

    st.markdown(
        "<h6 style='text-align: center; color: gray; font-size: 9px;'>© 2024 Rayen Ghali et Sid Ahmed Selouani. Tous droits réservés. Ce projet a bénéficié du soutien technique du Centre Anselme Chiasson de la bibliothèque Champlain de l'Université de Moncton, d'un financement conjoint de Mitacs et du Service expérientiel de l'Université de Moncton et d'un soutien administratif de Assomption. </h6>",
        # "<h6 style='text-align: center; color: gray; font-size: 9px;'>© 2024 Rayen Ghali. Travail réalisé sous la supervision de Sid Ahmed Selouani. Ce projet a bénéficié du financement conjoint de Mitacs et du Service expérientiel de l'Université de Moncton, soutien administratif de Assomption. Tous droits réservés.</h6>",
        unsafe_allow_html=True,
    )

prompt = st.chat_input("Message ChatAcadien...")

if DEBUGGING:

    @st.fragment
    def save_chat_logs():
        pass

else:

    @st.fragment
    def save_chat_logs():
        try:
            client = MongoClient(
                get_env_variable("MONGO_URI"), server_api=ServerApi("1")
            )
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


voyageai_embeddings = VoyageAIEmbeddings(model="voyage-3")


# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# _filter = LLMListwiseRerank.from_llm(llm, top_n=2)
def gen_create_custom_retriever_tool(index_name, top_n, description, embeddings_model):

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\n---\n",
        ],
        chunk_overlap=20,
        length_function=len,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7000, separators=["\n---\n"], chunk_overlap=2000, length_function=len
    )

    fs = LocalFileStore("./store_location_c")
    store = create_kv_docstore(fs)
    # store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        # parent_splitter=parent_splitter,
    )

    compressor = VoyageAIRerank(model="rerank-2", top_k=top_n)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[
            compressor,
            # _filter
        ]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

    retriever_tool = create_retriever_tool(
        compression_retriever,
        index_name,
        description,
    )

    return retriever_tool


def create_custom_retriever_tool(index_name, k, top_n, description, embeddings_model):
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    compressor = VoyageAIRerank(model="rerank-2", top_k=top_n)
    # compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[
            compressor,
            # _filter
        ]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
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
    k=3,
    top_n=2,
    description="Pour les questions relatives à la ceaac (tarifs, Horaires, politiques de consultation des archives), vous devez utiliser cet outil.",
    embeddings_model=voyageai_embeddings,
)


ceaac_faq_tool = create_custom_retriever_tool(
    index_name="ceaac-questions-frequemment-posees-index",
    k=12,
    top_n=3,
    description="Cet outil contient des questions fréquemment posées avec les réponses suggérées par le centre CEAAC. Utilisez cet outil en parallèle avec les autres outils.",
    embeddings_model=voyageai_embeddings,
)


genealogie_retriever_tool = gen_create_custom_retriever_tool(
    # index_name="genealogie-acadienne-index",
    # index_name="genealogie-acadienne-index-cwp", #with parents split to 7000 chars each, -c is the full parents (from one paragraph until the next para)
    index_name="genealogie-acadienne-index-c",
    top_n=2,
    description="Pour les questions relatives à la généalogie et aux familles acadiennes, vous devez utiliser cet outil. Les informations étant sensibles, assurez-vous de vérifier l'exactitude des noms, sachant que différentes personnes peuvent avoir le même nom. Demandez, si nécessaire, la possibilité d'obtenir plus d'informations. Ne répondez pas sans justification.",
    embeddings_model=voyageai_embeddings,
)

search_kwargs = {
    "count": 3,
    "summary": True,
    # "country": "CA",
}
search = BraveSearch.from_api_key(
    api_key=get_env_variable("BRAVE_API_KEY"),
    search_kwargs=search_kwargs,
)
# search = TavilySearchResults(max_results=3)

if st.session_state["years_limit"]:
    start_year = st.session_state["years_limit"]
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")  # Current date
    freshness_param = f"{start_date}to{end_date}"
    search_kwargs = {
        "count": 3,
        "summary": True,
        # "country": "CA",
        "freshness": freshness_param,
    }

    search = BraveSearch.from_api_key(
        api_key=get_env_variable("BRAVE_API_KEY"),
        search_kwargs=search_kwargs,
    )

search.description = "Un moteur de recherche web utile pour répondre à des questions sur des événements actuels ou des sujets spécifiques, comme le patrimoine acadien (histoire de l'Acadie, recettes, cuisine ou musique acadienne...). L'entrée doit être une requête de recherche."

tools = [
    search,
    ceaac_faq_tool,
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


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"Vous êtes un assistant virtuel du Centre d'études acadiennes Anselme-Chiasson (CEAAC). Répondez dans la même langue que l'utilisateur en anglais ou en français. Vos réponses doivent etre courtes et concises. Vous avez accès à des outils qui vous fournissent des informations spécifiques sur le centre. Pour les questions qui nécessitent un appel d'outil, effectuez toujours un appel simultané à l'outil ceaac-questions-frequemment-posees-index. Ne mentionnez pas quel outil vous utilisez. Si vous n'êtes pas en mesure de répondre à la demande de l'utilisateur, orientez-le selon le sujet vers l'adresse e-mail appropriée en vous référant à ce dictionnaire : {'; '.join(f'{key}: {value}' for key, value in subject_to_email.items())}. Retournez toujours à l'utilisateur vos sources, incluant le lien, l'auteur, le titre et la date de publication.",
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


@st.fragment
def escape_dollar_signs(input_text):
    return re.sub(r"(?<!\\)\$", r"\\$", input_text)


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


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="Images/avataruser.png"):
        st.write(prompt)


@st.fragment
async def process_events(model_name=None):
    # Use passed model_name or default from session state
    current_model = model_name or st.session_state.get(
        "DEFAULT_MODEL_NAME", "claude-3-5-sonnet-latest"
    )

    if "claude" in current_model.lower():
        model = ChatAnthropic(
            model=current_model,
            temperature=0,
            max_tokens=8096,
            timeout=None,
            max_retries=2,
            streaming=True,
        )
    else:
        model = ChatOpenAI(
            openai_api_key=get_env_variable("OPENROUTER_API_KEY"),
            openai_api_base=get_env_variable("OPENROUTER_BASE_URL"),
            model_name=current_model,
            temperature=0,
            max_tokens=8096,
            timeout=None,
            max_retries=2,
            streaming=True,
        )

    agent = create_tool_calling_agent(model, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        return_intermediate_steps=False,
    )

    accumulated_text = ""
    placeholder = st.empty()
    async for event in agent_executor.astream_events(
        {"input": st.session_state.messages[-1]["content"]}, version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                if "openai" in current_model.lower():
                    text = content
                else:
                    text = content[0].get("text", "")

                if text:
                    accumulated_text += escape_dollar_signs(text)
                    message_placeholder.empty()
                    message_placeholder.write(accumulated_text)
                    st.session_state["accumulated_text"] = escape_dollar_signs(
                        accumulated_text
                    )
        if event["event"] == "on_tool_end":
            st.session_state["accumulated_text"] = ""
            accumulated_text = ""
            message_placeholder.empty()


@st.fragment
async def generate_response():
    max_retries = 2
    retry_count = 0

    # Define model fallback order
    models = [
        st.session_state.get("DEFAULT_MODEL_NAME", "claude-3-5-sonnet-latest"),
        (
            "openai/gpt-4o"
            if st.session_state.get("DEFAULT_MODEL_NAME") == "claude-3-5-sonnet-latest"
            else "claude-3-5-sonnet-latest"
        ),
    ]
    # print(f"Attempting with models: {models}")

    while retry_count < max_retries:
        try:
            current_model = models[retry_count]
            print(f"Attempting with model: {current_model}")
            await process_events(model_name=current_model)
            break  # Exit loop if successful
        except Exception as e:
            retry_count += 1
            print(
                f"Error with {current_model}: {e}. Retrying {retry_count}/{max_retries}..."
            )
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Optional delay between retries

    if retry_count == max_retries:
        print("Max retries reached. Could not complete the task with any model.")


if DEBUGGING:

    if st.session_state.messages[-1]["role"] != "assistant":
        st_callback = StreamlitCallbackHandler(
            st.chat_message("assistant", avatar="Images/avatarchat.png"),
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            max_thought_containers=4,
        )
        response = agent_executor.invoke(  # TODO since definition of agent is now inside funtion need to redefine it also here
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
        message_placeholder = st.empty()
        st.session_state["feedback_container"] = False
        feedback_container = st.container()
else:
    if st.session_state.messages[-1]["role"] != "assistant":

        with st.chat_message("assistant", avatar="Images/avatarchat.png"):
            message_placeholder = st.empty()
            st.session_state["feedback_container"] = False
            feedback_container = st.container()
            feedback_container.empty()
        asyncio.run(generate_response())
        message = {"role": "assistant", "content": st.session_state["accumulated_text"]}
        st.session_state.messages.append(message)


# @st.fragment
def rerun_last_question():
    st.session_state["messages"].pop(-1)


@st.fragment
def log_feedback():
    st.toast(shown_strings["feedback"], icon=":material/thumbs_up_down:")


# TODO make it not save if DEBUGGING is True
if DEBUGGING:

    @st.fragment
    def save_to_db(feedback_msg):
        pass

else:

    @st.fragment
    def save_to_db(feedback_msg):
        log_feedback()
        try:
            client = MongoClient(
                get_env_variable("MONGO_URI"), server_api=ServerApi("1")
            )
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
    cols_dimensions = [7, 7, 100]
    col0, col1, col2 = st.columns(cols_dimensions, gap="medium")
    col0.button(
        ":material/autorenew:",
        on_click=rerun_last_question,
        key="rerun_last_question",
    )
    with col1.popover(":material/thumb_down:"):
        n_feedback()

    save_chat_logs()


if (
    len(st.session_state.messages) > 1
    and st.session_state.messages[-1]["role"] == "assistant"
    and not st.session_state["feedback_container"]
):
    st.session_state["feedback_container"] = True
    with feedback_container:
        feedback()


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
