import ollama
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber
import re

# Constants
JSON_PATH = "dataset.json"  # Fixed JSON path
PDF_PATH = "IHEC-PE-2018-2019-Campagne-Juillet-2020.pdf"  # Fixed PDF path
MODEL = "llama3.2"  # Change to the actual model you're using

# Initialize SentenceTransformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_dataset(json_path):
    """
    Load the JSON dataset into a list of (question, answer) for direct matching.
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    dataset = []
    if isinstance(data, dict) and "dataset" in data:
        for item in data["dataset"]:
            if "question" in item and "answer" in item:
                dataset.append((item["question"], item["answer"]))
    return dataset

def extract_json_text(json_path):
    """
    Extract 'question' and 'answer' fields and concatenate them into a single string
    for embedding-based context.
    """
    text = ""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "dataset" in data:
        for item in data["dataset"]:
            if "question" in item and "answer" in item:
                text += f"Q: {item['question']} A: {item['answer']} "
    return text.strip()

def extract_pdf_text(pdf_path):
    """
    Extract text from all pages of a PDF file.
    """
    if not os.path.exists(pdf_path):
        return ""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def combine_sources(json_path, pdf_path):
    """
    Combine text from the JSON and PDF files into a single string for chunking and embedding.
    """
    json_text = extract_json_text(json_path)
    pdf_text = extract_pdf_text(pdf_path)
    combined_text = f"{json_text}\n{pdf_text}"
    return combined_text.strip()

def chunk_text(text, chunk_size=500):
    """
    Split text into chunks of 'chunk_size' words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def build_embeddings(chunks):
    """
    Build embeddings for each chunk using the SentenceTransformer.
    """
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    return embeddings

def store_embeddings_in_faiss(embeddings):
    """
    Store embeddings in a FAISS index for similarity search.
    """
    embeddings_cpu = embeddings.cpu().detach().numpy()
    dimension = embeddings_cpu.shape[1]
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = np.ascontiguousarray(embeddings_cpu, dtype='float32')
    index.add(embeddings_np)
    return index

def generate_query_embedding(query):
    """
    Generate an embedding for the query text.
    """
    return embedder.encode(query, convert_to_tensor=True)

def search_relevant_context(query, index, chunks, threshold=1.0):
    """
    Search the FAISS index for the most relevant chunk.
    If the top match distance is above 'threshold', return None.
    """
    query_embedding = generate_query_embedding(query)
    query_embedding_cpu = query_embedding.cpu().detach().numpy().reshape(1, -1)
    D, I = index.search(query_embedding_cpu, k=1)
    if D[0][0] > threshold:
        return None
    return chunks[I[0][0]]

def build_and_store_index(json_path, pdf_path):
    """
    Combine text from sources, chunk it, embed it, and build a FAISS index.
    """
    combined_text = combine_sources(json_path, pdf_path)
    chunks = chunk_text(combined_text)
    embeddings = build_embeddings(chunks)
    index = store_embeddings_in_faiss(embeddings)
    return index, chunks

def build_prompt_with_context(query, index, chunks, dataset, image_path=None):
    """
    Build the prompt for Ollama's chat:
      1) Check if there's an exact match in dataset.json -> return that answer.
      2) Otherwise, find the best chunk from FAISS if relevant.
      3) If no relevant context, respond 'Je ne sais pas.'
    """
    # 1) Exact match check in the dataset
    for q, a in dataset:
        if query.lower() == q.lower():
            # Build a direct answer from the JSON
            return {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Tu es EKO, un assistant IA expert pour l'IHEC Carthage en Tunisie. "
                            "Réponds uniquement en français. Utilise les réponses exactes du fichier JSON "
                            "lorsque la question correspond mot pour mot. Si la question est hors contexte ou n'est pas dans le JSON, "
                            "réponds 'Je ne sais pas.' Pour les questions sur des données personnelles : "
                            "En tant qu'IA, je ne dois pas posséder des données personnelles. Ne mentionne jamais tes sources."
"N'oubliez pas que vous aidez les étudiants, il est donc essentiel de répondre de manière formelle, bienveillante et détaillée. Prenez le temps d'élaborer davantage dans vos réponses, en adoptant un ton charmant et encourageant pour les accompagner au mieux."
                        )
                    },
                    {"role": "system", "content": f"Réponse directe du dataset: {a}"},
                    {"role": "user", "content": query}
                ]
            }

    # 2) If no exact match, search for relevant chunk in the combined text
    relevant_context = search_relevant_context(query, index, chunks)
    if not relevant_context:
        # 3) No relevant info -> 'Je ne sais pas.'
        return {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Tu es EKO, un assistant IA expert pour l'IHEC Carthage en Tunisie. "
                        "Réponds uniquement en français. Utilise le PDF uniquement pour le contexte de l'IHEC Carthage "
                        "si ce n’est pas dans le JSON. Si aucune info n’existe, dis: Je ne sais pas."
                        "Si la question est sur des données personnelles: En tant qu'IA, je ne dois pas posséder des données personnelles."
"N'oubliez pas que vous aidez les étudiants, il est donc essentiel de répondre de manière formelle, bienveillante et détaillée. Prenez le temps d'élaborer davantage dans vos réponses, en adoptant un ton charmant et encourageant pour les accompagner au mieux."
                    )
                },
                {"role": "system", "content": "Aucun contexte trouvé."},
                {"role": "user", "content": query}
            ]
        }

    # 2b) Found a relevant chunk from PDF/JSON combined text
    messages = [
        {
            "role": "user",
            "content": (
                "Tu es EKO, un assistant IA expert pour l'IHEC Carthage en Tunisie. "
                "Réponds uniquement en français. Utilise les réponses exactes du fichier JSON si disponible. "
                "Sinon, utilise le PDF comme contexte sur l'IHEC Carthage. Ne mentionne jamais tes sources textuellement. "
                "Ne réponds pas si tu n'as pas d'informations. Dans ce cas dis: Je ne sais pas. "
                "Pour les questions sur des données personnelles: En tant qu'IA, je ne dois pas posséder des données personnelles."
"N'oubliez pas que vous aidez les étudiants, il est donc essentiel de répondre de manière formelle, bienveillante et détaillée. Prenez le temps d'élaborer davantage dans vos réponses, en adoptant un ton charmant et encourageant pour les accompagner au mieux."
            )
        },
        {"role": "system", "content": f"Contexte pertinent: {relevant_context}"}
    ]
    if image_path:
        messages.append({"role": "system", "content": f"Image fournie: {os.path.basename(image_path)}"})
    messages.append({"role": "user", "content": query})

    return {"model": MODEL, "messages": messages}

def sanitize_data(text):
    """
    Removes/masks phone numbers, email addresses, and capitalized two-word names.
    Adjust or expand patterns as needed.
    """
    # Phone patterns
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  
        r'\b\d{2,4}[-.]?\d{2,4}[-.]?\d{2,4}\b'
    ]
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    for pattern in phone_patterns:
        text = re.sub(pattern, '*****', text)
    text = re.sub(email_pattern, '*****', text)

    # Mask capitalized two-word names, e.g., "John Doe"
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '*****', text)

    return text

def load_history():
    """
    Load conversation history from a JSON file (if it exists).
    """
    try:
        with open('history.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_history(history):
    """
    Save conversation history to a JSON file.
    """
    with open('history.json', 'w', encoding='utf-8') as file:
        json.dump(history, file, indent=4)

def main():
    print("Welcome to the AI Chat Assistant for IHEC Carthage!")
    history = load_history()
    print(f"Using fixed JSON file: {JSON_PATH}")
    print(f"Using fixed PDF file: {PDF_PATH}")
    print("Type 'exit' to quit.")

    # Check file existence
    if not os.path.exists(JSON_PATH) and not os.path.exists(PDF_PATH):
        print("Error: Neither the JSON nor the PDF file was found.")
        return

    # Build index for FAISS searching
    index, chunks = build_and_store_index(JSON_PATH, PDF_PATH)

    # Load dataset into memory for exact matches
    dataset = load_dataset(JSON_PATH)

    print("Index built! Ready to answer queries.\n")

    # Optional image
    image_path = None
    add_image = input("Upload an optional image for context? (yes/no): ").strip().lower()
    if add_image == "yes":
        image_path = input("Enter the image path: ").strip()
        if not os.path.exists(image_path):
            print("File not found. Continuing without an image.")
            image_path = None
        else:
            print("Image file loaded!")

    # Chat loop
    while True:
        query = input("\nYour Query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        sanitized_query = sanitize_data(query)
        prompt = build_prompt_with_context(sanitized_query, index, chunks, dataset, image_path)

        try:
            response = ollama.chat(**prompt)
            ai_reply = response['message']['content']
            print("\nAI Response:")
            print(ai_reply)
            
            # Save to history
            history.append({'query': sanitized_query, 'response': ai_reply})
            save_history(history)

        except Exception as e:
            print(f"Error communicating with Ollama: {e}")

if __name__ == "__main__":
    main()
