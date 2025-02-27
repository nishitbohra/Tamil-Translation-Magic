# 🌟 Tamil Translation Magic ✨

A Streamlit-based application that provides **real-time English-to-Tamil translation** using **Facebook's M2M100 model**. This app also includes **romanized Tamil output**, **cultural context**, **batch translation**, and **text-to-speech** functionality.

## 🚀 Features
- 🔄 **Real-time English to Tamil Translation**
- 🔠 **Romanized Tamil Output**
- 🔊 **Text-to-Speech for Tamil Translations**
- 📚 **Batch Translation from CSV Files**
- 🎨 **Cultural Learning Mode**
- 🎯 **Interactive UI with Streamlit**

## 🏗️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/nishitbohra/Tamil-Translation-Magic.git
cd Tamil-Translation-Magic
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit App
```bash
streamlit run Machinetrans.py
```
### 🔧 Dependencies
1. streamlit
2. torch
3. transformers
4. indic-transliteration
5. pandas
6. plotly
7. gtts
8. PIL
9. numpy
To install all dependencies:

```bash
pip install streamlit torch transformers indic-transliteration pandas plotly gtts pillow numpy
```

### 🛠️ How It Works
- **Single Sentence Mode 📝:** Enter text and get instant Tamil translation with pronunciation and audio.
- **Batch Translation Mode 📊:** Upload a CSV file, translate multiple sentences, and download the results.
- **Cultural Learning Mode 🎭:** Learn common Tamil phrases based on categories like greetings, food, festivals, etc.
