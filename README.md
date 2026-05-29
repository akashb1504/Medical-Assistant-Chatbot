# AI Medical Assistant Chatbot

![Medical Assistant Chatbot](chatbot-ui.jpeg)

An end-to-end AI-powered Medical Assistant Chatbot built using FastAPI, Streamlit, LangChain, Pinecone, Groq, and FastEmbed.

The application allows users to upload medical PDF documents and ask medical questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

# Live Demo

Streamlit

```text
https://medical-assistant-chat-bot.streamlit.app/
```
```

GitHub Repository

```text
https://github.com/akashb1504/Medical-Assistant-Chatbot.git
```

---

# Overview

* Upload medical PDF documents
* Generate embeddings using FastEmbed
* Store vectors in Pinecone
* Retrieve relevant document chunks
* Generate AI responses using Groq LLM
* Fully deployed using Render and Streamlit Cloud

---

# Tech Stack

## Frontend

* Streamlit

## Backend

* FastAPI
* Uvicorn

## AI / RAG

* LangChain
* Groq
* FastEmbed
* Pinecone

## Deployment

* Render
* Streamlit Cloud

---

# Quick Setup

## 1. Clone Repository

```bash
git clone https://github.com/akashb1504/Medical-Assistant-Chatbot.git
```

---

# Backend Setup

## Navigate to Server

```bash
cd Medical-Assistant-Chatbot/server
```

## Create Virtual Environment

```bash
uv venv
```

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
uv pip install -r requirements.txt
```

## Create `.env` File

```env
GROQ_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
```

## Run FastAPI Backend

```bash
uvicorn main:app --reload --port 8000
```

---

# Frontend Setup

## Navigate to Client

```bash
cd ../client
```

## Create Virtual Environment

```bash
uv venv
```

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```


## Install Dependencies

```bash
uv pip install -r requirements.txt
```

## Run Streamlit Frontend

```bash
streamlit run app.py
```


