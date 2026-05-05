import json
import streamlit as st
from connect_all import connect_all

with open('app_config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

st.set_page_config(page_title="Card Generator", layout="wide")


with st.sidebar:

    st.header("Generation Settings")
    if 'decks_data' not in st.session_state:
        st.session_state.decks_data = data['deck_name'] + [""]
    if 'terms_data' not in st.session_state:
        st.session_state.terms_data = data['front_card_name'] + [""]


    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<br><b>Deck name</b>", unsafe_allow_html=True)

    with col2:
        st.write("**Front termin name**")

    for i in range(len(st.session_state.decks_data)):
        with col1:

            st.text_input(
                label="Deck name",
                label_visibility="collapsed",
                value=st.session_state.decks_data[i],
                key=f"deck_{i}"
            )
        with col2:
            st.text_input(
                label="Termin name",
                label_visibility="collapsed",
                value=st.session_state.terms_data[i],
                key=f"term_{i}"
            )


    if st.button("Add row"):
        st.session_state.decks_data.append("")
        st.session_state.terms_data.append("")
        st.rerun()


    st.markdown("---")

    gen_images = st.checkbox("Generate Images?", help="Not recommended for CPU or GPU with low VRAM", value=data['generate_images'])
    gen_audio = st.checkbox("Generate audio?", value=data['generate_audio'])
    offload_cpu = st.checkbox("Offload CPU", help="Only for generating Images. If generating is too long (10+ minutes for 5 cards) try to off/on this setting", value=data['offload_cpu'])

    if st.button("Save settings", use_container_width=True):

        all_decks = []
        all_terms = []

        for i in range(len(st.session_state.decks_data)):
            all_decks.append(st.session_state.get(f"deck_{i}"))
            all_terms.append(st.session_state.get(f"term_{i}"))

        data['deck_name'] = list(filter(lambda x: True if x else False, all_decks))
        data['front_card_name'] = all_terms[:len(data['deck_name'])]
        data['generate_images'] = gen_images
        data['generate_audio'] = gen_audio
        data['offload_cpu'] = offload_cpu

        with open('app_config.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        st.success("Settings applied!")


st.title("Please make sure Ollama and Anki with Anki Connect are running")
main_deck = st.text_input(label="Deck in which you want to add cards")

user_text = st.text_area("Enter your text here:", height=200, placeholder="Start typing...")

value = st.slider("Select value:", min_value=1, max_value=25, value=10)

notification_placeholder = st.empty()

if st.button("Generate cards", use_container_width=True):

    notification_placeholder.empty()

    if data['necessary_models']:
        with st.spinner('Generating cards, this may take some time...'):
            connect_all(
                user_text,
                data['deck_name'],
                data['front_card_name'],
                value,
                data['generate_images'] and data['image_models'], 
                data['generate_audio'] and data['TTS_models'],
                main_deck,
                data['offload_cpu']
            )

        notification_placeholder.success('All cards generated, check Anki!')
            
            
    else:
        notification_placeholder.error('You don\'t have required models, run first_launch.py and download the necessary models')
