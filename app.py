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


def query(payload):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {gemma_api_key}",
            # "HTTP-Referer": f"{YOUR_SITE_URL}", # Optional, for including your app on openrouter.ai rankings.
            # "X-Title": f"{YOUR_APP_NAME}", # Optional. Shows in rankings on openrouter.ai.
        },
        data=json.dumps(
            {
                "model": "google/gemma-7b-it:free",
                # "model": "mistralai/mistral-7b-instruct:free",  # Optional
                "messages": [{"role": "user", "content": payload}],
                "top_p": 1,
                "temperature": 0.75,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.4,
                "repetition_penalty": 1,
                "top_k": 0,
            }
        ),
    )
    return response.json()


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        # {"role": "assistant", "content": "How may I assist you today?"}
        {"role": "assistant", "content": "Comment puisse-je vous aidez?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        # {"role": "assistant", "content": "How may I assist you today?"}
        {"role": "assistant", "content": "Comment puisse-je vous aidez?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_response(prompt_input):
    # if language == "fr":
    starting_string = "Vous êtes un assistant. Vous ne répondez pas en tant que 'Utilisateur' ou ne prétendez pas être 'Utilisateur'. Pour chaque identifie d'abord son language et repond dans le même language. Vous ne répondez qu'une seule fois en tant qu' 'Assistant'.\n\n"
    # string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    # if len(st.session_state.messages) > 2:
    # string_dialogue = ""
    string_dialogue = starting_string
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    history = string_dialogue + "\n"

    prompt_text = history + "\n prompt : " + prompt_input

    if len(prompt_text) > 8192:
        prompt_text = starting_string + history[-2048:] + "\n prompt : " + prompt_input

    st.write(history, len(history))
    st.write(prompt_text)

    output = query(prompt_text)
    return output["choices"][0]["message"]["content"]


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
