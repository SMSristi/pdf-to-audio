import streamlit as st
from PyPDF2 import PdfReader
from gtts import gTTS
import tempfile
import time

st.set_page_config(page_title="PDF to Audio", page_icon="🎧")

st.title("🎧 PDF to Audio Converter")
st.markdown("Upload a PDF file, and this app will convert it into an audiobook using text-to-speech.")

# Upload the PDF
pdf_file = st.file_uploader("📄 Choose a PDF file", type=["pdf"])

if pdf_file:
    reader = PdfReader(pdf_file)
    full_text = ""
    num_pages = len(reader.pages)

    # Estimate time
    if num_pages > 1:
        est_time = num_pages * 2  # ~2 seconds per page (estimated)
        st.info(f"⏱️ This PDF has {num_pages} pages. Estimated processing time: ~{est_time} seconds.")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"

        # Update progress bar
        progress = int((i + 1) / num_pages * 100)
        progress_bar.progress(progress)
        status_text.text(f"🔄 Processing page {i + 1} of {num_pages}...")

        time.sleep(0.1)  # simulate progress (optional)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Button to convert text to audio
    if st.button("🎙️ Convert to Audio"):
        if full_text.strip() == "":
            st.warning("⚠️ No readable text found in the PDF.")
        else:
            with st.spinner("🎧 Generating audio... please wait..."):
                tts = gTTS(text=full_text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tts.save(tmp_file.name)
                    audio_file = tmp_file.name
            st.success("✅ Audio generated successfully!")
            st.audio(audio_file, format="audio/mp3")
