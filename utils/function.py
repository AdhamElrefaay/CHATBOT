from typing import Generator, List, Dict
import PyPDF2
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_together import ChatTogether


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def read_system_prompt(file_path: str) -> str:
    """
    Reads the system prompt from a file.
    
    Args:
        file_path (str): Path to the system prompt file.
    
    Returns:
        str: Content of the system prompt file.
    """
    with open(file_path, "r") as file:
        return file.read()


def get_response_from_llm(
    user_question: str,
    chat_history: List[Dict[str, str]],
    system_prompt: str,
    api_key: str,
    session_id: str = "default_session",
) -> Generator[str, None, None]:
    """
    Generates a streaming response from the LLM using a system prompt, chat history, and user question.
    
    Args:
        user_question (str): The user's question.
        chat_history (List[Dict[str, str]]): List of previous chat interactions.
        system_prompt (str): The system prompt to guide the LLM.
        api_key (str): API key for the LLM.
        session_id (str): Unique session ID for memory management.
    
    Yields:
        str: Chunks of the LLM's response.
    """
    # Initialize LLM with streaming enabled
    llm = ChatTogether(
        model="meta-llama/Llama-3-70b-chat-hf",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
        streaming=True,  # Enable streaming
    )

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Create the chain
    chain = prompt | llm

    # Initialize memory
    history = ChatMessageHistory()
    for interaction in chat_history:
        history.add_user_message(interaction["user"])
        history.add_ai_message(interaction["bot"])

    # Wrap the chain with memory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history"
    )

    # Stream the response
    for chunk in chain_with_memory.stream(
        {"input": user_question},
        config={"configurable": {"session_id": session_id}}
    ):
        yield chunk.content