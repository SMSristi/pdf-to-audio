import streamlit as st
from PyPDF2 import PdfReader
from gtts import gTTS
import tempfile

st.set_page_config(page_title="PDF to Audio", page_icon="🎧")

st.title("🎧 PDF to Audio Converter")
st.markdown("Upload a PDF file, and this app will convert it into an audiobook.")

pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if pdf_file:
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    if st.button("Convert to Audio"):
        if full_text.strip() == "":
            st.warning("❗ No readable text found in the PDF.")
        else:
            tts = gTTS(text=full_text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                audio_file = tmp_file.name
            st.audio(audio_file, format="audio/mp3")
            st.success("✅ Audio generated successfully!")
