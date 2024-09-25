# utils
import uuid
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Literal, Any, Union

# langchain \ langgraph
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant():
    """
    A class representing an agent that handles invoking a runnable object and managing retries based on the response.

    The Assistant class is responsible for:
    - Storing a runnable instance (which contains the logic for processing agent requests).
    - Determining if a retry is needed based on the response.
    - Updating the state to prompt a valid response if retries are necessary.
    - Handling the workflow of invoking the runnable until a valid response is received.
    """

    def __init__(self, runnable: Runnable):
        """
        Initializes the Assistant with a runnable instance.

        Args:
            runnable (Runnable): An instance of a runnable object used to invoke the agent's workflow.
        """
        self.runnable = runnable

    def should_retry(self, response):
        """
        Determines whether the response warrants a retry.
        A retry is needed if there are no tool calls in the response or if the content is empty 
        or missing text in a valid output.

        Args:
            response (object): The response returned by the runnable's invocation.

        Returns:
            bool: True if a retry should be attempted, False otherwise.
        """
        return (
            not response.tool_calls and
            (not response.content or 
             (isinstance(response.content, list) and not response.content[0].get('text')))
        )

    def update_state_for_retry(self, state: AgentState):
        """
        Updates the agent's state for retry by appending a message to guide the response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            state (AgentState): The modified state ready for a retry.
        """
        state["messages"].append("respond with a real output")
        return state

    def __call__(self, state: AgentState, config: RunnableConfig):
        """
        Invokes the runnable and handles retry logic until a valid response is returned.

        Args:
        state (AgentState): The current state passed to the runnable for invocation.
        config (RunnableConfig): Configuration for the runnable invocation.

        Returns:
            dict: The final response containing valid messages from the agent's invocation.
        """
        response = self.runnable.invoke(state)
        
        while self.should_retry(response):
            state = self.update_state_for_retry(state)
            response = self.runnable.invoke(state)
    
        return {"messages": response}


def assistant_answer(app:Runnable, example: dict):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    messages = app.invoke({"messages": [example["input"]]}, config)
    
    return {
        "response": messages["messages"][-1].content, 
        "messages": messages
    }

