import streamlit as st
from PyPDF2 import PdfReader
from gtts import gTTS
import io
import textwrap
import time

# --- Performance Optimizations ---

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    The @st.cache_data decorator ensures this function only runs once per file.
    """
    reader = PdfReader(pdf_file)
    full_text = ""
    num_pages = len(reader.pages)
    
    # Use st.status for a cleaner progress indicator
    with st.status(f"Processing {num_pages} pages...", expand=True) as status:
        for i, page in enumerate(reader.pages):
            st.write(f"Reading page {i + 1}...")
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            time.sleep(0.05) # Small delay for UX
        status.update(label="✅ Text extracted successfully!", state="complete")
        
    return full_text

def split_text(text, max_chars=4000):
    """
    Splits text into chunks suitable for gTTS.
    """
    return textwrap.wrap(text, max_chars, break_long_words=True, replace_whitespace=False)

# --- Streamlit App UI ---

st.set_page_config(page_title="PDF to Audio", page_icon="🎧", layout="wide")

st.title("🎧 PDF to Audio Converter")
st.markdown("Upload a PDF and this app will convert it to audio, chunk by chunk.")

# 1. File Uploader
pdf_file = st.file_uploader("📄 Choose a PDF file", type=["pdf"])

if pdf_file:
    # 2. Extract Text (uses cached function)
    full_text = extract_text_from_pdf(pdf_file)

    if not full_text.strip():
        st.warning("⚠️ No readable text was found in the PDF.")
    else:
        st.subheader("📖 Extracted Text")
        st.text_area("Review the text extracted from your PDF:", full_text, height=200)

        # 3. Convert to Audio
        st.subheader("🎙️ Generated Audio")
        
        text_chunks = split_text(full_text)
        
        if not text_chunks:
            st.info("The document appears to be empty.")
        else:
            with st.spinner(f"Generating audio for {len(text_chunks)} chunk(s)..."):
                for i, chunk in enumerate(text_chunks):
                    try:
                        # Generate audio for each chunk in memory
                        tts = gTTS(text=chunk, lang='en')
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0) # Go to the beginning of the in-memory file

                        st.write(f"**Part {i + 1} / {len(text_chunks)}**")
                        st.audio(audio_fp, format="audio/mp3")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to generate audio for chunk {i + 1}. Error: {e}")

                st.success("✅ All audio parts generated successfully!")
