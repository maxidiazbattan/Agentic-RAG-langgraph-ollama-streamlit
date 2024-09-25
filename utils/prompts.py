# langchain | langgraph
from langchain_core.prompts import ChatPromptTemplate

# Create the primary assistant prompt template
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
         "system",
         "You are a helpful assistant tasked with answering user questions. "
         "You have access to two tools: retrieve, and web_search. "
         "For any user questions about documents, use the retrive tool to get information for a vectorstore."
         "For any other questions, such as questions about current events, use the web_search to get information from the web. "
         "Please check your tools available before answer any questions",
            
        ),
        ("placeholder", "{messages}"),
    ]
)
