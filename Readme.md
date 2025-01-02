# Multimodal Chatbot with RAG and Audio I/O

This project implements a multimodal chatbot that leverages Retrieval-Augmented Generation (RAG) and audio input/output capabilities. The chatbot can interact with users through text, voice, and visual inputs, providing answers based on uploaded documents, images, and spoken questions.

## Features

- **Document-based Q&A**: Upload PDFs, CSVs, or provide an ArXiv paper ID to ask questions and get summaries.
- **Visual Question Answering**: Upload an image and ask questions about its content.
- **Speech Recognition**: Use your microphone to ask questions.
- **Text-to-Speech**: Listen to the chatbot's responses.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/HazemAbdelghafar/Multimodal-RAG-Chatbot.git
    cd Multimodal-RAG-Chatbot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a ***.env*** file in the root directory.
    - Add your Google API key and other necessary environment variables:
        ```
        GEMINI_API_KEY=your_gemini_api_key
        ```

5. Place your Google service account key file in the root directory and name it **service-account-key.json**.

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Use the sidebar to navigate between "Chat with Documents" and "Visual Question Answering".

### Chat with Documents

- Upload PDF or CSV files, or enter an ArXiv paper ID.
- Ask questions or request summaries based on the uploaded documents.
- Use the microphone button to ask questions via speech.
- Listen to the chatbot's responses using the text-to-speech button.

### Visual Question Answering

- Upload an image (JPG or PNG).
- Ask questions about the content of the image.

## File Structure

- [app.py](./app.py): Main application file for the Streamlit interface.
- [audio_module.py](./audio_module.py): Handles speech recognition and text-to-speech functionalities.
- [cv_module.py](./cv_module.py): Manages visual question answering using the BLIP model.
- [llm_module.py](./llm_module.py): Implements the logic for the language model and retrieval-augmented generation.
## Dependencies

- Streamlit
- SpeechRecognition
- Pygame
- gTTS
- transformers
- langchain
- arxiv
- PIL
- dotenv

## Acknowledgements

- [LangChain](https://github.com/langchain/langchain)
- [Hugging Face](https://huggingface.co/)
- [Google Cloud](https://cloud.google.com/)
- [Streamlit](https://streamlit.io/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
