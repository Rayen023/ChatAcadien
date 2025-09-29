*ChatAcadien is an LLM-based conversational agent designed to provide interactive access to Acadian genealogical records dating from 1700-1900. The application is accessible at [https://chatacadien.ca/](https://chatacadien.ca/).*

## Overview

ChatAcadien serves as a specialized knowledge base assistant for researchers and individuals interested in Acadian genealogy and history. The system provides access to genealogical records, historical information, and institutional documentation from the Centre d'études acadiennes Anselme-Chiasson (CEAAC) at the University of Moncton.

## Funding & Acknowledgments

This project benefited from technical support from the Anselme Chiasson Centre of the Champlain Library at the University of Moncton, joint funding from Mitacs and the University of Moncton's Experiential Learning Service, and administrative support from Assomption.

## Technical Architecture

### Language Models
- Primary: OpenAI GPT-4.1 (via OpenRouter)
- Fallback: Anthropic Claude-3.7-Sonnet

### Tech Stack
- **Frontend Framework**: Streamlit
- **Vector Database**: Pinecone
- **Embeddings**: VoyageAI (voyage-3 model)
- **Orchestration**: LangChain
- **Data Storage**:
  - MongoDB for conversation and feedback logging
  - Pinecone for document storage
- **Search Capabilities**: BraveSearch API

### Document Processing Pipeline
- **Text Splitting Strategy**:
  - Parent documents: RecursiveCharacterTextSplitter (7000 characters)
  - Child documents: RecursiveCharacterTextSplitter (1024 characters with 20-character overlap)
- **Retrieval Approach**: ParentDocumentRetriever for context preservation
- **Reranking**: VoyageAI rerank-2 model for improved results relevance

### Specialized Retrievers
The system implements multiple dedicated retrieval tools:
1. `genealogie-acadienne-index-c`: For historical Acadian family records
2. `ceaac-general-info-index`: For institutional policies and information
3. `ceaac-questions-frequemment-posees-index`: For FAQ

## Features

- **Bilingual Support**: Fully functional in both French and English
- **Streaming Responses**: Real-time output generation
- **Conversation Memory**: Maintains context throughout the interaction
- **Tool Calling**: Structured information retrieval through specialized tools
- **Parent-Child Document Linking**: Preserves broader context of source documents
- **User Feedback System**: Collects and logs user feedback for continuous improvement
- **Web Search Integration**: Supplements knowledge with current information when needed
