# AI Medical Assistant Chatbot

![Medical Assistant Chatbot](chatbot-ui.png)

An end-to-end AI-powered Medical Assistant Chatbot built using FastAPI, Streamlit, LangChain, Pinecone, Groq, and FastEmbed.

The application allows users to upload medical PDF documents and ask medical questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

# Live Demo

Frontend (Streamlit)

```text
https://medical-assistant-chat-bot.streamlit.app/
```

Backend API (Render)

```text
https://medical-assistant-chatbot-1.onrender.com
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

# Local Reproduction Steps

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

### Linux / Mac

```bash
source .venv/bin/activate
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

Backend runs on:

```text
http://127.0.0.1:8000
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

### Linux / Mac

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
uv pip install -r requirements.txt
```

## Run Streamlit Frontend

```bash
streamlit run app.py
```

Frontend runs on:

```text
http://localhost:8501
```

---

# Project Summary

Built an end-to-end AI Medical Assistant Chatbot using FastAPI, Streamlit, LangChain, Pinecone, Groq, and FastEmbed. Implemented Retrieval-Augmented Generation (RAG) with PDF upload, semantic search, vector embeddings, and AI-generated responses. Optimized deployment for free-tier cloud hosting and deployed the application using Render and Streamlit Cloud.

---

# Author

Akash B
