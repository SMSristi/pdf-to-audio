import streamlit as st
from PyPDF2 import PdfReader
from gtts import gTTS
import tempfile
import time

import textwrap

def split_text(text, max_chars=4000):
    return textwrap.wrap(text, max_chars)

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
                audio_path = None
                try:
                    # Split text into chunks (gTTS can't handle very long strings)
                    chunks = split_text(full_text)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        for chunk in chunks:
                            tts = gTTS(text=chunk, lang='en')
                            tts.write_to_fp(tmp_file)
                            time.sleep(1)  # ⏱️ Delay between requests to avoid 429 error
                        audio_path = tmp_file.name


                    st.success("✅ Audio generated successfully!")
                    st.audio(audio_path, format="audio/mp3")
                except Exception as e:
                    st.error("❌ Failed to generate audio. Try again with a smaller or simpler PDF.")
                    st.exception(e)

