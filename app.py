import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import io
import textwrap

# --- Core Functions for PDF Processing and AI ---

@st.cache_data
def extract_text_from_pdf(pdf_file_contents):
    """
    Extracts text from a PDF using PyMuPDF (fitz), which is more robust.
    """
    with fitz.open(stream=pdf_file_contents, filetype="pdf") as doc:
        full_text = "".join(page.get_text() for page in doc)
    return full_text

@st.cache_data
def chunk_text_for_reader(text, max_chars=4000):
    """
    Splits text into large chunks suitable for the book reader feature.
    """
    return textwrap.wrap(text, max_chars, break_long_words=True, replace_whitespace=False)

@st.cache_data
def chunk_text_for_rag(text):
    """
    Splits text into smaller, overlapping chunks for the RAG pipeline.
    This helps the AI find more accurate context.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache_resource
def setup_rag_pipeline(chunks):
    """
    Creates vector embeddings and builds the FAISS index for searching.
    This is the core of the RAG system.
    """
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embedder

@st.cache_data
def search_in_pdf(_index, _embedder, question, _chunks, k=3):
    """
    Searches the FAISS index to find the most relevant text chunks for a question.
    Note: Underscores on args tell Streamlit not to hash them, saving time.
    """
    question_embedding = _embedder.encode([question])
    _, I = _index.search(np.array(question_embedding).astype('float32'), k)
    return [_chunks[i] for i in I[0]]

# --- Streamlit App UI ---

st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI PDF Assistant")
st.markdown("Listen to your document or ask it questions directly.")

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    # --- Setup Phase (runs once per file upload) ---
    with st.spinner("Analyzing your document... This may take a moment."):
        file_contents = uploaded_file.getvalue()
        
        # 1. Extract text with the better library
        full_text = extract_text_from_pdf(file_contents)
        
        # 2. Chunk text for both features
        reader_chunks = chunk_text_for_reader(full_text)
        rag_chunks = chunk_text_for_rag(full_text)
        
        # 3. Set up the RAG pipeline (heavy lifting is cached)
        rag_index, embedder_model = setup_rag_pipeline(rag_chunks)
    
    st.success("Document analyzed successfully! You can now use the features below.")
    
    # --- UI Tabs for Features ---
    read_tab, chat_tab = st.tabs(["📖 Read Aloud (Book Reader)", "💬 Chat with PDF (AI Q&A)"])

    # --- Book Reader Feature ---
    with read_tab:
        st.header("Listen to the Full Document")
        st.write("Click the button below to generate audio for the entire document, part by part.")
        
        if st.button("🎙️ Generate Full Audio"):
            if not full_text.strip():
                st.warning("⚠️ No readable text was found in the PDF.")
            else:
                with st.spinner(f"🎧 Generating {len(reader_chunks)} audio parts..."):
                    for i, chunk in enumerate(reader_chunks):
                        try:
                            # Using 'bn' for Bangla, change if needed
                            tts = gTTS(text=chunk, lang='bn')
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            st.write(f"**[translate:অংশ] {i + 1} / {len(reader_chunks)}**")
                            st.audio(audio_fp, format="audio/mp3")
                        except Exception as e:
                            st.error(f"❌ Failed to generate audio for part {i+1}. Error: {e}")
                    st.success("✅ All audio parts generated!")

    # --- AI Chat Feature ---
    with chat_tab:
        st.header("Ask a Question About Your Document")
        st.markdown("Your question will be answered by an AI based on the document's content.")
        
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("Searching for answers..."):
                # 1. Find relevant context
                relevant_chunks = search_in_pdf(rag_index, embedder_model, question, rag_chunks)
                context = "\n---\n".join(relevant_chunks)
                
                # 2. Get answer from LLM
                llm = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
                result = llm(question=question, context=context)
                answer_text = result['answer']

                st.subheader("💡 Answer:")
                st.write(f"> {answer_text}")

                # 3. Generate audio for the answer
                with st.spinner("Generating audio for the answer..."):
                    try:
                        tts_answer = gTTS(text=answer_text, lang='en')
                        audio_fp_answer = io.BytesIO()
                        tts_answer.write_to_fp(audio_fp_answer)
                        audio_fp_answer.seek(0)
                        st.audio(audio_fp_answer, format="audio/mp3")
                    except Exception as e:
                        st.error(f"❌ Failed to generate audio for the answer. Error: {e}")

