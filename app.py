import streamlit as st
from google.cloud import vision
from pdf2image import convert_from_bytes
from transformers import pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import io
import textwrap

# --- Cloud Vision OCR for Bengali ---

@st.cache_data
def extract_text_with_cloud_vision(pdf_file_contents, api_key):
    """
    Uses Google Cloud Vision API for highly accurate Bengali text extraction.
    """
    # Set up the client with API key
    client = vision.ImageAnnotatorClient(
        client_options={"api_key": api_key}
    )
    
    # Convert PDF to images
    images = convert_from_bytes(pdf_file_contents)
    
    full_text = ""
    for i, img in enumerate(images):
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Create vision image object
        image = vision.Image(content=img_byte_arr.getvalue())
        
        # Perform OCR with Bengali language hint
        response = client.document_text_detection(
            image=image,
            image_context={"language_hints": ["bn"]}  # Bengali language hint
        )
        
        # Extract text
        if response.full_text_annotation.text:
            full_text += response.full_text_annotation.text + "\n"
        
        st.write(f"✓ Processed page {i+1}/{len(images)}")
    
    return full_text

@st.cache_data
def chunk_text_for_reader(text, max_chars=4000):
    return textwrap.wrap(text, max_chars, break_long_words=True, replace_whitespace=False)

@st.cache_data
def chunk_text_for_rag(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < text_length:
            last_period = max(chunk.rfind('.'), chunk.rfind('।'), chunk.rfind('?'), chunk.rfind('!'))
            if last_period > chunk_size - 200:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

@st.cache_resource
def setup_rag_pipeline(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embedder

@st.cache_data
def search_in_pdf(_index, _embedder, question, _chunks, k=3):
    question_embedding = _embedder.encode([question])
    _, I = _index.search(np.array(question_embedding).astype('float32'), k)
    return [_chunks[i] for i in I[0]]

# --- Streamlit App UI ---

st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI PDF Assistant for Bengali Documents")
st.markdown("Upload a Bengali PDF to listen to it or ask questions about it.")

# API Key input (secure)
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google Cloud Vision API Key", type="password")
    st.caption("Get your free API key from: [Google Cloud Console](https://console.cloud.google.com/)")

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file and google_api_key:
    with st.spinner("📚 Analyzing your document with Google Cloud Vision..."):
        file_contents = uploaded_file.getvalue()
        
        try:
            # Extract text using Google Cloud Vision
            full_text = extract_text_with_cloud_vision(file_contents, google_api_key)
            
            # Chunk text for both features
            reader_chunks = chunk_text_for_reader(full_text)
            rag_chunks = chunk_text_for_rag(full_text)
            
            # Set up RAG pipeline
            rag_index, embedder_model = setup_rag_pipeline(rag_chunks)
            
            st.success("✅ Document analyzed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("💡 Make sure your API key is correct and the Vision API is enabled.")
            st.stop()
    
    # Preview extracted text
    with st.expander("📄 Preview Extracted Text"):
        st.text_area("First 500 characters:", full_text[:500], height=150)
    
    # UI Tabs
    read_tab, chat_tab = st.tabs(["📖 Read Aloud", "💬 Chat with PDF"])

    with read_tab:
        st.header("Listen to the Full Document")
        
        if st.button("🎙️ Generate Full Audio"):
            if not full_text.strip():
                st.warning("⚠️ No text found.")
            else:
                with st.spinner(f"🎧 Generating {len(reader_chunks)} audio parts..."):
                    for i, chunk in enumerate(reader_chunks):
                        try:
                            tts = gTTS(text=chunk, lang='bn')
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.write(f"**Part {i + 1} / {len(reader_chunks)}**")
                            st.audio(audio_fp, format="audio/mp3")
                        except Exception as e:
                            st.error(f"❌ Error for part {i+1}: {e}")
                    st.success("✅ All audio generated!")

    with chat_tab:
        st.header("Ask a Question About Your Document")
        
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("🔍 Searching..."):
                relevant_chunks = search_in_pdf(rag_index, embedder_model, question, rag_chunks)
                context = "\n---\n".join(relevant_chunks)
                
                try:
                    llm = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
                    result = llm(question=question, context=context)
                    answer_text = result['answer']

                    st.subheader("💡 Answer:")
                    st.write(f"> {answer_text}")

                    tts_answer = gTTS(text=answer_text, lang='bn')
                    audio_fp_answer = io.BytesIO()
                    tts_answer.write_to_fp(audio_fp_answer)
                    audio_fp_answer.seek(0)
                    st.audio(audio_fp_answer, format="audio/mp3")
                except Exception as e:
                    st.error(f"❌ Could not generate answer: {e}")

elif uploaded_file and not google_api_key:
    st.warning("⚠️ Please enter your Google Cloud Vision API Key in the sidebar.")
