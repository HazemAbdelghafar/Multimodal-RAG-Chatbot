import os
import arxiv
from PIL import Image
import streamlit as st
from llm_module import pipeline, invoke_llm
from audio_module import recognize_speech_from_mic, speak_text
from cv_module import initialize_visual_answering, invoke_visual
from langchain_community.document_loaders import PyPDFLoader, CSVLoader


@st.cache_resource
def process_files(files, arxiv_id = None):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the temp directory exists

    documents = []
    if arxiv_id:
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
        paper.download_pdf('./temp', 'arxiv_paper.pdf')
        loader = PyPDFLoader('./temp/arxiv_paper.pdf')
        documents.extend(loader.load())
        os.remove('./temp/arxiv_paper.pdf')
    
    if files:
        for file in files:
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file.read())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif file.name.endswith(".csv"):
                loader = CSVLoader(temp_path)
            else:
                st.error("Unsupported file type.")
                continue
            
            documents.extend(loader.load())
            os.remove(temp_path)  # Clean up temporary file

    
    return documents

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'user_input' not in st.session_state:
    st.session_state.user_input = None

st.set_page_config(page_title="Multimodal Chatbot with RAG and Audio I/O", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Chat with Documents", "Visual Question Answering"])

if page == "Chat with Documents":

    st.title("LLM-powered Chatbot with RAG")
    st.header("Chat with your documents")
    st.sidebar.header("Upload PDFs, CSVs, or enter an ArXiv paper ID for processing.")
    st.markdown("This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions and summarize content from PDFs, CSVs, and arXiv papers.")

    uploaded_files = st.sidebar.file_uploader("Upload PDF or CSV files", type=["pdf", "csv"], accept_multiple_files=True)
    arxiv_id = st.sidebar.text_input("Enter ArXiv ID")

    documents = process_files(uploaded_files,arxiv_id) if uploaded_files or arxiv_id else []        
        
    if documents: 
        chain = pipeline(documents)
    else:
        if 'chain' not in st.session_state or st.session_state.chain is None:
            chain = pipeline()
        else:
            chain = st.session_state.chain

    st.session_state.chain = chain

    if chain:
        # Create columns for microphone button and input
        col1, col2 = st.columns([1, 9])
        listener = st.empty()
        # Microphone Button
        with col1:
            if st.button("🎤", help="Use microphone input"):
                listener.write("Listening...")
                recognized_text = recognize_speech_from_mic()
                if recognized_text and recognized_text.strip():
                    st.session_state.user_input = recognized_text
                else:
                    user_input = None
                    st.session_state.user_input = None
                listener.empty()

        # Input Field
        with col2:
            user_input = st.text_input("Ask a question or request a summary:", value=st.session_state.user_input)

        if user_input:
            with st.spinner("Generating response..."):
                response = invoke_llm(chain, user_input, st.session_state.messages)
            # Update conversation history
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})
            if response and response.strip():
                st.session_state.messages.append({"role": "bot", "content": response})
        # Display response and TTS button
        if st.session_state.messages:
            # Get the last bot response
            last_response = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "bot"), None)
            if last_response:
                # Add TTS button to read the last response aloud
                if st.button("🔊", help="Read response aloud"):
                    speak_text(last_response)

        # Display conversation history
        st.markdown("### Conversation History")
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**User:** {message['content']}")
            else:
                st.markdown(f"**Chatbot:** {message['content']}")
        else:
            st.info("Upload documents to enable the chatbot.")

elif page == "Visual Question Answering":
    st.sidebar.header("Image Input for Visual Question Answering")
    image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    processor, blip_model = initialize_visual_answering()
    
    if image_file:
        try:
            # Load and display the image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", width=500)
            
            # Accept a question about the image
            visual_question = st.text_input("Ask a question about the image:")
            if visual_question:
                try:
                    response = invoke_visual(processor, blip_model, image, visual_question)
                    st.write(f"Visual Answer: {response}")
                except Exception as e:
                    st.error(f"Error during visual question answering: {e}")
        except Exception as e:
            st.error(f"Invalid image format: {e}")
    else:
        st.info("Please upload an image to ask a question.")

        