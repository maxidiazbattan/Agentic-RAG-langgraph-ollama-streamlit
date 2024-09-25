# utils
import os 
from typing import Annotated, List, Dict, Literal, Any, Union
from typing_extensions import TypedDict

# langchain | langgraph
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable, RunnableLambda, RunnableWithFallbacks

# streamlit
import streamlit as st


def load_documents(query:str, load_max_docs:int, model_name:str='nomic-embed-text'):
    """
    Function to load documents based on the user query.
    """
    # print('---LOADING DOCS---')
    st.session_state.log += "---LOADING DOCS---" + "\n\n"
    st.session_state.placeholder.markdown("---LOADING DOCS---")
    loader = ArxivLoader(query=query, load_max_docs=load_max_docs)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len
    )

    new_docs = text_splitter.split_documents(documents=docs)
    doc_strings = [doc.page_content for doc in new_docs]
    embeddings = OllamaEmbeddings(model=model_name)
    db = Chroma.from_documents(new_docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever


@tool
async def retrieve(state):
    """
    Function to retrieve documents.
    """
    # print('---RETRIEVE---')
    st.session_state.log += "---RETRIEVE---" + "\n\n"
    st.session_state.placeholder.markdown("---RETRIEVE---")

    retriever = load_documents(state, 3)
    
    documents = retriever.get_relevant_documents(state)
    return {'messages': [state, documents]}


@tool
async def web_search(state):
    """
    Function to call the web search.
    """
    # print('---WEB SEARCH---')
    st.session_state.log += "---WEB SEARCH---" + "\n\n"
    st.session_state.placeholder.markdown("---WEB SEARCH---")

    try:
        tool = DuckDuckGoSearchResults()
        resut = tool.invoke(state)
   
        return {'messages': [resut]}
    except Exception as e:
        print(e)


def handle_tool_error(state) -> dict:
    """
    Handles errors related to tool calls in the agent's state by returning a formatted error message.

    This function retrieves the error from the agent's state and iterates over the tool calls from the most recent message.
    For each tool call, it generates an error message and associates it with the corresponding tool call ID, prompting the user
    to fix the mistake.

    Args:
        state (dict): The current state of the agent, which includes information about tool calls and errors.

    Returns:
        dict: A dictionary containing error messages associated with each tool call.
    """

    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )



