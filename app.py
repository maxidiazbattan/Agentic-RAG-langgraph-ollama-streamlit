# tools
import glob

# utils
from utils.tools import load_documents, retrieve, web_search, create_tool_node_with_fallback
from utils.prompts import  primary_assistant_prompt
from utils.assistant import AgentState, Assistant

# langchain | langgraph
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# streamlit
import streamlit as st

import asyncio


tools = [retrieve, web_search]

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    stream=True
)

assistant = primary_assistant_prompt | llm.bind_tools(tools)

tool_node = ToolNode(tools)


@st.cache_resource
def create_graph():
    """
    Creates and compiles a state graph for managing the interaction between an assistant agent and tool nodes.

    This function builds a state graph using a `StateGraph` builder, adding nodes for the assistant and tools, 
    and defining the flow between them based on conditions. The assistant node represents the primary agent 
    interaction, while the tools node handles external tool invocations.

    The graph defines:
        - An initial edge from the start of the workflow to the assistant node.
        - Conditional edges from the assistant node to either the tools node or the end of the workflow, based on 
        whether the assistant's response is a tool call.
        - A loop from the tools node back to the assistant node for continued interaction after a tool call.

    Returns:
        StateGraph: A compiled state graph for execution of the assistant-agent workflow.
    """

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("assistant", Assistant(assistant))
    workflow.add_node("tools", create_tool_node_with_fallback(tools)) # for the tools

    # Add edges
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges(
        "assistant",
        # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
        # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    workflow.add_edge("tools", "assistant")

    return workflow.compile()


async def run_workflow(inputs):

    """
    Asynchronously runs the workflow based on the provided inputs and updates the Streamlit session state.

    This function initiates a trial count in the session state and displays a status message during the workflow execution. 
    It asynchronously invokes the workflow stored in the session state with the given inputs, and once the process is 
    complete, it updates the UI to display the result and marks the status as finished.

    Args:
        inputs (dict): The inputs to be passed to the workflow for processing.

    Returns: 
        None
    """

    st.session_state.number_trial = 0
    with st.status(label="**---INIT---**", expanded=True, state="running") as st.session_state.status:
        st.session_state.placeholder = st.empty()
        value = await st.session_state.workflow.ainvoke(inputs)

    st.session_state.placeholder.empty()
    st.session_state.message_placeholder = st.empty()
    st.session_state.status.update(label="**---FINISH---**", state="complete", expanded=False)
    st.session_state.message_placeholder.markdown(value)


def main() -> None:

    st.title("Arxiv retriever and Web search Agent 📑")

    if 'log' not in st.session_state:
        st.session_state.log = ""

    if 'status_container' not in st.session_state:
        st.session_state.status_container = st.empty()

    if not hasattr(st.session_state, "workflow"):
        graph = create_graph()
        st.session_state.workflow = graph
    
    if prompt := st.chat_input("Insert your query"):
        with st.chat_message("user"):
            st.markdown(prompt)

        inputs = {"messages": [prompt]}

        asyncio.run(run_workflow(inputs))

if __name__=='__main__':
    main()
