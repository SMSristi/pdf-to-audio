import streamlit as st
from PyPDF2 import PdfReader
from gtts import gTTS
import io
import textwrap
import time

# --- Performance Optimizations ---

@st.cache_data
def extract_text_from_pdf(pdf_file_contents):
    """
    Extracts text from the content of an uploaded PDF file.
    This function is cached and contains NO Streamlit UI elements.
    It returns the full text and the number of pages.
    """
    pdf_stream = io.BytesIO(pdf_file_contents)
    reader = PdfReader(pdf_stream)
    full_text = ""
    num_pages = len(reader.pages)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text, num_pages

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
    # Get the file's content once
    file_contents = pdf_file.getvalue()
    
    # 2. Extract Text with UI feedback in the main app body
    full_text, num_pages = extract_text_from_pdf(file_contents)
    
    # Estimate time (this UI element was preserved)
    if num_pages > 1:
        est_time = num_pages * 2  # ~2 seconds per page (estimated)
        st.info(f"⏱️ This PDF has {num_pages} pages. Estimated processing time: ~{est_time} seconds.")
    
    # Check if text was found
    if not full_text.strip():
        st.warning("⚠️ No readable text was found in the PDF.")
    else:
        st.subheader("📖 Extracted Text")
        st.text_area("Review the text extracted from your PDF:", full_text, height=200)

        # Button to convert text to audio (preserved from your original logic)
        if st.button("🎙️ Convert to Audio"):
            st.subheader("🎙️ Generated Audio")
            
            text_chunks = split_text(full_text)
            
            if not text_chunks:
                st.info("The document appears to be empty.")
            else:
                with st.spinner(f"🎧 Generating audio for {len(text_chunks)} chunk(s)... please wait..."):
                    for i, chunk in enumerate(text_chunks):
                        try:
                            # Using 'bn' for Bangla, as discussed
                            tts = gTTS(text=chunk, lang='bn')
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)

                            # Preserved the English "Part" label as requested
                            st.write(f"**Part {i + 1} / {len(text_chunks)}**")
                            st.audio(audio_fp, format="audio/mp3")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to generate audio for chunk {i + 1}. Error: {e}")
                            st.exception(e) # Preserved detailed error logging

                    st.success("✅ Audio generated successfully!")

