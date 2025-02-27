import streamlit as st
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import pandas as pd
import plotly.express as px
from gtts import gTTS
import io
import base64
from PIL import Image
import numpy as np
import time

# Set page config
st.set_page_config(
    page_title="Tamil Translation Magic ✨",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
    }
    .translation-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_1.2B"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, model, tokenizer):
    tokenizer.src_lang = "en"
    encoded = tokenizer(text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id("ta"),
            max_length=128,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True
        )
    
    tamil_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    romanized = transliterate(tamil_text, sanscript.TAMIL, sanscript.IAST)
    return tamil_text, romanized

def create_audio(text, lang='ta'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_b64 = base64.b64encode(audio_buffer.getvalue()).decode()
        return f'<audio controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
    except:
        return None

def get_cultural_context(text):
    # Simple keyword-based context (can be expanded)
    contexts = {
        'temple': 'Temples (கோயில்) are central to Tamil culture and architecture.',
        'food': 'Tamil cuisine is known for its use of rice, lentils, and unique spice blends.',
        'festival': 'Festivals like Pongal and Tamil New Year are important cultural celebrations.',
        'music': 'Carnatic music is the classical music tradition of South India.',
        'dance': 'Bharatanatyam is the classical dance form of Tamil Nadu.',
    }
    
    relevant_contexts = []
    for keyword, context in contexts.items():
        if keyword.lower() in text.lower():
            relevant_contexts.append(context)
    return relevant_contexts

def main():
    st.title("🌟 Tamil Translation Magic ✨")
    st.markdown("### Bringing Tamil Language & Culture to Life")
    
    # Initialize session state
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    
    # Sidebar
    st.sidebar.title("⚙️ Settings & Tools")
    translation_mode = st.sidebar.selectbox(
        "Choose Translation Mode",
        ["Single Sentence", "Batch Translation", "Cultural Learning"]
    )
    
    # Load model
    with st.spinner("Loading translation model... 🔄"):
        model, tokenizer = load_model()
    
    if translation_mode == "Single Sentence":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Enter English Text 📝")
            input_text = st.text_area("", height=150)
            
            if st.button("✨ Translate"):
                if input_text:
                    with st.spinner("Translating... 🎯"):
                        tamil_text, romanized = translate_text(input_text, model, tokenizer)
                        
                        # Store in history
                        st.session_state.translation_history.append({
                            'english': input_text,
                            'tamil': tamil_text,
                            'romanized': romanized
                        })
        
        with col2:
            st.markdown("### Tamil Translation 🎯")
            if input_text and 'tamil_text' in locals():
                st.markdown("#### Tamil Script:")
                st.markdown(f"<div class='translation-box'>{tamil_text}</div>", unsafe_allow_html=True)
                
                st.markdown("#### Romanized Version:")
                st.markdown(f"<div class='translation-box'>{romanized}</div>", unsafe_allow_html=True)
                
                # Audio generation
                st.markdown("#### 🔊 Listen to the Translation:")
                audio_html = create_audio(tamil_text)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
                
                # Cultural context
                contexts = get_cultural_context(input_text)
                if contexts:
                    st.markdown("#### 📚 Cultural Context:")
                    for context in contexts:
                        st.info(context)
    
    elif translation_mode == "Batch Translation":
        st.markdown("### Batch Translation 📚")
        uploaded_file = st.file_uploader("Upload CSV file with 'English' column", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'English' not in df.columns:
                st.error("CSV must contain 'English' column!")
            else:
                if st.button("🚀 Translate Batch"):
                    progress_bar = st.progress(0)
                    translated_data = []
                    
                    for i, row in df.iterrows():
                        tamil_text, romanized = translate_text(row['English'], model, tokenizer)
                        translated_data.append({
                            'English': row['English'],
                            'Tamil': tamil_text,
                            'Romanized': romanized
                        })
                        progress_bar.progress((i + 1) / len(df))
                    
                    result_df = pd.DataFrame(translated_data)
                    st.markdown("### Results:")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="translated_data.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    else:  # Cultural Learning
        st.markdown("### Cultural Learning Mode 🎨")
        categories = ["Greetings", "Food", "Festivals", "Family", "Numbers"]
        selected_category = st.selectbox("Choose a category to learn:", categories)
        
        # Pre-defined phrases for each category
        phrases = {
            "Greetings": [
                "Hello (Vanakkam)",
                "How are you? (Eppadi irukeenga?)",
                "Thank you (Nandri)",
                "Welcome (Varaverppu)"
            ],
            "Food": [
                "Rice (Saatham)",
                "Curry (Kari)",
                "Water (Thanneer)",
                "Spicy (Kaaram)"
            ]
            # Add more categories and phrases
        }
        
        if selected_category in phrases:
            for phrase in phrases[selected_category]:
                with st.expander(phrase):
                    english = phrase.split(" (")[0]
                    tamil_text, romanized = translate_text(english, model, tokenizer)
                    st.markdown(f"**Tamil Script:** {tamil_text}")
                    st.markdown(f"**Romanized:** {romanized}")
                    audio_html = create_audio(tamil_text)
                    if audio_html:
                        st.markdown("**Listen:**")
                        st.markdown(audio_html, unsafe_allow_html=True)
    
    # Translation History
    if st.session_state.translation_history:
        st.sidebar.markdown("### 📚 Translation History")
        for i, item in enumerate(st.session_state.translation_history[-5:]):  # Show last 5
            with st.sidebar.expander(f"Translation {len(st.session_state.translation_history)-i}"):
                st.write("English:", item['english'])
                st.write("Tamil:", item['tamil'])
                st.write("Romanized:", item['romanized'])

if __name__ == "__main__":
    main()