# Agent with Langgraph, Ollama and Streamlit

This repository showcases a test of the Llama 3.1 model tool-calling capabilities through a streamlined Agentic Retrieval-Augmented Generation (RAG) application with a Streamlit frontend. The application is built using LangGraph and Ollama, enabling the agent to retrieve relevant information from documents and generate coherent, contextually appropriate responses.

In addition to document-based retrieval, the agent can search the web for answers to queries unrelated to the retrieved documents, ensuring comprehensive coverage across various topics by leveraging internet-based tools. This setup demonstrates the model efficiency in both document-focused and web-based information retrieval, making it adaptable for diverse use cases.

If you want, you can test this app in lightning studios via this [link](https://lightning.ai/maxidiazbattan/studios/langgraph-agenticrag-with-streamlit), or locally following the steps below.

### 1. [Install](https://github.com/ollama/ollama?tab=readme-ov-file) ollama and pull models

On linux
```shell
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the LLM you'd like to use and the embedding model:

```shell
ollama pull llama3.1

ollama pull nomic-embed-text
```

### 2. Create a virtual environment

```shell
python -m venv venv
source venv/bin/activate
```

### 3. Install libraries

```shell
pip install -r requirements.txt
```

### 4. Run the agent

```shell
python app.py
```
