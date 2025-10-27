import streamlit as st
import easyocr
from pdf2image import convert_from_bytes
from transformers import pipeline, VitsModel, AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import scipy.io.wavfile as wavfile
import io
import textwrap


# --- EasyOCR for Bengali (FREE - No API Key Needed) ---

@st.cache_resource
def load_ocr_reader():
    '''Load EasyOCR reader (cached to avoid reloading)'''
    return easyocr.Reader(['bn', 'en'], gpu=False)


@st.cache_data
def extract_text_with_easyocr(pdf_file_contents):
    '''
    Uses EasyOCR for free Bengali text extraction.
    '''
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


# --- UPGRADED: Meta MMS-TTS for Bengali (Better quality than gTTS) ---

@st.cache_resource
def load_tts_model():
    '''Load Meta's MMS-TTS model for Bengali (much better quality than gTTS)'''
    model = VitsModel.from_pretrained("facebook/mms-tts-ben")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    return model, tokenizer


def generate_audio_mms(text):
    '''
    Generate audio using Meta MMS-TTS (higher quality than gTTS)
    Returns: audio bytes in WAV format
    '''
    try:
        model, tokenizer = load_tts_model()
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Convert to numpy array and save to bytes
        waveform = output.squeeze().cpu().numpy()
        
        # Create audio file in memory
        audio_buffer = io.BytesIO()
        # MMS-TTS outputs at 16kHz sample rate
        wavfile.write(audio_buffer, rate=16000, data=(waveform * 32767).astype(np.int16))
        audio_buffer.seek(0)
        
        return audio_buffer
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None


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


# --- UPGRADED: Multilingual Embeddings (Better for Bengali) ---

@st.cache_resource
def setup_rag_pipeline(chunks):
    '''Setup RAG with multilingual embeddings (supports Bengali better)'''
    # Using multilingual model instead of English-only
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embedder


@st.cache_data
def search_in_pdf(_index, _embedder, question, _chunks, k=3):
    question_embedding = _embedder.encode([question])
    _, I = _index.search(np.array(question_embedding).astype('float32'), k)
    return [_chunks[i] for i in I[0]]


# --- UPGRADED: BanglaBERT for Question Answering (Bengali-specific) ---

@st.cache_resource
def load_qa_model():
    '''Load BanglaBERT model for Bengali question answering'''
    model_name = "csebuetnlp/banglabert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return qa_pipeline


# --- Streamlit App UI ---

st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI PDF Assistant for Bengali Documents")
st.markdown("Upload a Bengali PDF to listen to it or ask questions about it.")
st.info("🆕 **Upgraded with**: Meta MMS-TTS (better audio quality) + BanglaBERT (accurate Bengali QA)")

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
            
            # Set up RAG pipeline with multilingual embeddings
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
        st.caption("🎵 Using Meta MMS-TTS for natural-sounding Bengali speech")
        
        if st.button("🎙️ Generate Full Audio"):
            if not full_text.strip():
                st.warning("⚠️ No text found.")
            else:
                with st.spinner(f"🎧 Generating {len(reader_chunks)} audio parts..."):
                    for i, chunk in enumerate(reader_chunks):
                        try:
                            # Generate audio using Meta MMS-TTS
                            audio_buffer = generate_audio_mms(chunk)
                            
                            if audio_buffer:
                                st.write(f"**Part {i + 1} / {len(reader_chunks)}**")
                                st.audio(audio_buffer, format="audio/wav")
                        except Exception as e:
                            st.error(f"❌ Error for part {i+1}: {e}")
                    st.success("✅ All audio generated!")

    with chat_tab:
        st.header("Ask a Question About Your Document")
        st.caption("🧠 Using BanglaBERT for accurate Bengali question answering")
        
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("🔍 Searching..."):
                # Retrieve relevant chunks
                relevant_chunks = search_in_pdf(rag_index, embedder_model, question, rag_chunks)
                context = "\n---\n".join(relevant_chunks)
                
                try:
                    # Load BanglaBERT QA model
                    qa_model = load_qa_model()
                    
                    # Get answer
                    result = qa_model(question=question, context=context)
                    answer_text = result['answer']
                    confidence = result['score']

                    st.subheader("💡 Answer:")
                    st.write(f"> {answer_text}")
                    st.caption(f"Confidence: {confidence:.2%}")

                    # Generate audio for the answer
                    st.write("🔊 **Listen to Answer:**")
                    audio_buffer = generate_audio_mms(answer_text)
                    if audio_buffer:
                        st.audio(audio_buffer, format="audio/wav")
                        
                except Exception as e:
                    st.error(f"❌ Could not generate answer: {e}")
                    st.write("**Relevant Context Found:**")
                    st.text_area("Context", context[:500], height=150)
