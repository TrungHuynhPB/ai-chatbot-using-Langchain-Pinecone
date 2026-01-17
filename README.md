# Chatbot Answering from Your Own Knowledge Base: Langchain, ChatGPT, Pinecone, and Streamlit
![main-·-Streamlit (1)](https://github.com/TrungHuynhPB/ai-chatbot-using-Langchain-Pinecone/blob/main/img/example.png)

## Deployment

#### 1. Clone Repository 

```bash
  git clone https://github.com/TrungHuynhPB/ai-chatbot-using-Langchain-Pinecone
```
```bash
  cd ai-chatbot-using-Langchain-Pinecone
```
#### 2. Install Dependencies

This project uses `uv` for dependency management.
Install uv with:
```bash
  pip install poetry, uv
```

Then install dependencies with:

```bash
  uv init
  uv sync (or uv add -r requiremnets.txt)
```


To activate the virtual environment:

 - For Windows:
```bash
  .venv\Scripts\activate
```
 - For macOS/Linux:
```bash
  source .venv/bin/activate
```
#### 4. Set Up Environment Variables

Copy the `env_example.txt` file to `.env` and fill in your API keys:

```bash
  cp env_example.txt .env
```

Edit `.env` with your credentials:
 - OpenAI API Key: [Get from OpenAI](https://platform.openai.com)
 - Pinecone API Key and Environment: [Get from Pinecone](https://app.pinecone.io)

#### 5. Replace your own documents in **data** folder

#### 6. Create Pinecone Index

Run the index creation script to set up your Pinecone index:
  uv run python -m  index_create <index_name>.


Example:
```bash
  uv run python -m  index_create langchain-chatbot
```

Make sure your Pinecone index matches Pinecone index_name and settings. For example:
   - **index_name = "langchain-chatbot"**
   - **Dimensions: 1536**

#### 7. Ingest and indexing your data (PDF)
```bash
  uv run python -m  indexing cryptocurrencies.pdf
```

#### 8. Replace your own OpenAI, Pinecone API Key and Pinecone environment in indexing.py, main.py & utils.py

#### 9. Run the web app
```bash
  uv run streamlit run main.py
```
