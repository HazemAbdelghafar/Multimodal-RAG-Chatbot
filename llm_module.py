import os
from copy import deepcopy
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account-key.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



def initialize_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings

def initialize_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.2,max_retries=2, api_key=GEMINI_API_KEY)
    return llm

def create_vectorstore(docs, embedder):
    if docs:
        return FAISS.from_documents(docs, embedder)
    return None

def promt_template():
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    system_prompt = """
    You are a helpful assistant tasked with answering questions based on the provided context. Follow these rules carefully:

    1. Only use the provided context to answer the question. Do not add information or assumptions beyond what is given.
    2. If the answer cannot be determined from the context, explicitly state: 
    "I don't know, but you can check other resources online."
    3. If the answer can be determined, provide it concisely, limited to a maximum of five sentences.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return prompt, contextualize_q_prompt

def create_conversational_chain(llm, retriever, prompt, contextualize_q_prompt):
    history_aware_retriever = create_history_aware_retriever(llm,retriever, contextualize_q_prompt)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return retrieval_chain

def pipeline(documents=None):
    llm = initialize_llm()
    
    embeddings = initialize_embeddings()
    
    vectorstore = create_vectorstore(documents, embeddings) if documents else None
    retriever = vectorstore.as_retriever() if vectorstore else None

    prompt, contextualize_q_prompt = promt_template()
    
    chain = create_conversational_chain(llm, retriever, prompt, contextualize_q_prompt) if retriever else llm
    return chain

def invoke_llm(chain, user_input:str, messages):
    history=[]
    
    for message in messages:
        if message["role"] == "user" and message['content']:
            user_query = message["content"]
            history.append(user_query)
        elif message["role"] == "bot" and message['content']:
            system_response = message["content"]
            history.append(system_response)

    try:
        if hasattr(chain,'model') and chain.model == "models/gemini-1.5-flash":
            llm_convo = deepcopy(history)
            llm_convo.append(user_input)
            response = chain.invoke(llm_convo)
        else:
            response = chain.invoke({"input": user_input, "chat_history": history})
                
        # Check if response is an AIMessage or a dictionary-like object
        if isinstance(response, AIMessage):
            final_response = response.content
        elif isinstance(response, dict) and "answer" in response:
            final_response = response["answer"].strip()
            
        return final_response
    except Exception as e:
        print(e)
        pass
