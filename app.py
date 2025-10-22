import streamlit as st
import easyocr
from pdf2image import convert_from_bytes
from transformers import pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import io
import textwrap


# --- EasyOCR for Bengali (FREE - No API Key Needed) ---

@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader (cached to avoid reloading)"""
    return easyocr.Reader(['bn', 'en'], gpu=False)


@st.cache_data
def extract_text_with_easyocr(pdf_file_contents):
    """
    Uses EasyOCR for free Bengali text extraction.
    """
    reader = load_ocr_reader()
    
    # Convert PDF to images
    images = convert_from_bytes(pdf_file_contents)
    
    full_text = ""
    for i, img in enumerate(images):
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Perform OCR (detail=0 returns only text, not coordinates)
        results = reader.readtext(img_array, detail=0, paragraph=True)
        
        # Join results
        page_text = " ".join(results)
        full_text += page_text + "\n"
        
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

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    with st.spinner("📚 Analyzing your document with EasyOCR..."):
        file_contents = uploaded_file.getvalue()
        
        try:
            # Extract text using EasyOCR (FREE)
            full_text = extract_text_with_easyocr(file_contents)
            
            # Chunk text for both features
            reader_chunks = chunk_text_for_reader(full_text)
            rag_chunks = chunk_text_for_rag(full_text)
            
            # Set up RAG pipeline
            rag_index, embedder_model = setup_rag_pipeline(rag_chunks)
            
            st.success("✅ Document analyzed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
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
