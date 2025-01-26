# EKO - IHEC Chatbot

EKO is an advanced chatbot designed to provide students and staff at IHEC (Institut des Hautes Études Commerciales) with reliable, contextual, and dynamic responses to their queries. By leveraging AI and a combination of structured and unstructured data, EKO aims to enhance the user experience and streamline access to institutional information. This project builds upon the foundational work described in the IHEC Data Processing Pipeline and represents a fully dynamic chatbot solution aligned with the objectives of that pipeline.
![image](https://github.com/user-attachments/assets/5459efcb-dec9-4011-ab01-c9e8b0e05f28)

---

## Features

1. **Dynamic Query Handling**:
   - Answers queries directly from a structured JSON dataset or provides context from institutional PDFs.
   - Processes natural language queries to identify relevant answers.

2. **Advanced Data Processing**:
   - Combines structured (JSON) and unstructured (PDF) data sources.
   - Creates embeddings for efficient semantic search using FAISS.

3. **Contextual AI**:
   - Utilizes **LLaMA 3.2** for generating responses in French.
   - Embeds relevant context from datasets and documents for accurate replies.

4. **Data Privacy**:
   - Sanitizes user queries to prevent the handling of personal data such as phone numbers or emails.

5. **Robust Architecture**:
   - Built using Flask for a lightweight and scalable web service.
   - Seamlessly integrates FAISS, SentenceTransformers, and Ollama for enhanced NLP capabilities.

---

## Models and Tools Used

### 1. **FAISS** (Facebook AI Similarity Search)
   - Indexes and retrieves relevant chunks of text from PDFs for semantic search.
   - Reduces latency and ensures high-accuracy context retrieval.

### 2. **LLaMA 3.2** (Response Generation)
   - Generates conversational, natural, and formal responses in French tailored to the IHEC audience.

### 3. **SentenceTransformers** (Embedding Model)
   - Uses `all-MiniLM-L6-v2` to encode text into semantic embeddings for efficient indexing and searching.

### 4. **Flask Framework**
   - Serves as the backend for hosting the chatbot interface and handling user interactions.

### 5. **PDFPlumber** (PDF Text Extraction)
   - Extracts text content from PDF documents, enabling the chatbot to provide responses based on institutional policies and historical data.

---

## Workflow

1. **Data Preparation**:
   - Extracts structured Q&A pairs from `dataset.json`.
   - Extracts and preprocesses text from PDF documents.

2. **FAISS Indexing**:
   - Combines JSON and PDF data into text chunks.
   - Embeds chunks using SentenceTransformers and stores them in a FAISS index for semantic retrieval.

3. **Dynamic Query Handling**:
   - Matches user queries against the dataset for direct answers.
   - If no exact match is found, searches FAISS for the most relevant context.

4. **Response Generation**:
   - Uses LLaMA 3.2 to formulate responses based on retrieved context.
   - Handles edge cases (e.g., no relevant data found) gracefully by responding "Je ne sais pas."

5. **Web Interface**:
   - Provides an intuitive interface for user interaction via Flask.

---

## Technical Importance

The development of EKO builds upon the foundation established by the **IHEC Data Processing Pipeline**. Key contributions include:

- **Dynamic Data Integration**: The pipeline’s structured outputs are utilized directly for chatbot responses, ensuring that the chatbot always reflects the latest institutional updates.
- **Semantic Search**: Advanced embeddings and FAISS indexing enable contextually relevant answers, even for nuanced queries.
- **Context-Rich Interaction**: Combines insights from PDFs and FAQs, allowing for deeper, more informed responses.
- **Scalable Architecture**: Modular design allows for easy updates as institutional needs evolve.

By aligning with the foundational objectives described in the IHEC Data Processing Pipeline README, this project exemplifies the synergy between data engineering and AI-driven conversational systems, delivering a truly dynamic chatbot experience.

---

## Deployment

### Prerequisites

1. **Python 3.8+**
2. **Dependencies**:
   - `flask`
   - `faiss`
   - `sentence-transformers`
   - `pdfplumber`
   - `ollama`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/eko-chatbot.git
   cd eko-chatbot
   ```

2. Prepare data files (`dataset.json`, `IHEC PDF`):
   - Place them in the project root.

3. Run the application:
   ```bash
   python app.py
   ```
4. Access the chatbot at:
   ```
   http://localhost:5000
   ```

---

## Outputs

1. **Real-Time Chat Responses**: Provides accurate, formal, and student-friendly answers in French.
2. **Error Handling**: Ensures robust fallback mechanisms for unsupported queries.
3. **Semantic Search Results**: Retrieves relevant document sections dynamically from FAISS.

---

## Future Enhancements

- Integration with additional data sources, such as IHEC’s live databases.
- Support for multilingual queries and responses.
- Advanced analytics to monitor chatbot performance and improve user satisfaction.

---
The database was renewed using this repo : https://github.com/cheedli/IHEC-SCRIPTS


The presentation link :https://www.canva.com/design/DAGdQRYDleI/7r3ub_NG5wUxhMrgTEXcmA/edit
