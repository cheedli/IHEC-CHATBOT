import os
import re
import json

from flask import Flask, request, jsonify, render_template
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# -------------------------------------------------------------------
# Constants / Config
# -------------------------------------------------------------------
JSON_PATH = "dataset.json"
PDF_PATH  = "IHEC-PE-2018-2019-Campagne-Juillet-2020.pdf"
MODEL     = "llama3.2" 

# -------------------------------------------------------------------
# Initialize embedding model
# -------------------------------------------------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------------------------------------------
# Helper Functions (FAISS, PDF, JSON)
# -------------------------------------------------------------------
def load_dataset(json_path):
    """Load Q&A from dataset.json into list of (question, answer)."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = []
    if isinstance(data, dict) and "dataset" in data:
        for item in data["dataset"]:
            q = item.get("question")
            a = item.get("answer")
            if q and a:
                dataset.append((q, a))
    return dataset

def extract_json_text(json_path):
    """Extract all Q/A from JSON as a single text."""
    if not os.path.exists(json_path):
        return ""
    text = ""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "dataset" in data:
        for item in data["dataset"]:
            q = item.get("question", "")
            a = item.get("answer", "")
            text += f"Q: {q} A: {a} "
    return text.strip()

def extract_pdf_text(pdf_path):
    """Extract text from all pages of the PDF."""
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
    """Combine JSON + PDF into one big string."""
    txt_json = extract_json_text(json_path)
    txt_pdf  = extract_pdf_text(pdf_path)
    return f"{txt_json}\n{txt_pdf}".strip()

def chunk_text(text, chunk_size=500):
    """Split text into word-chunks of size chunk_size."""
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_embeddings(chunks):
    """Embed each chunk via sentence-transformers."""
    return embedder.encode(chunks, convert_to_tensor=True)

def store_embeddings_in_faiss(embeddings):
    """Store embeddings in a FAISS index."""
    cpu_data  = embeddings.cpu().detach().numpy()
    dim       = cpu_data.shape[1]
    index     = faiss.IndexFlatL2(dim)
    contiguous = np.ascontiguousarray(cpu_data, dtype='float32')
    index.add(contiguous)
    return index

def generate_query_embedding(query):
    """Generate an embedding for the query."""
    return embedder.encode(query, convert_to_tensor=True)

def search_relevant_context(query, index, chunks, threshold=0.9):
    """
    Return best chunk from index if the distance is below threshold;
    otherwise return None.
    (Lowering threshold reduces the chance of picking unrelated text.)
    """
    q_emb = generate_query_embedding(query)
    q_cpu = q_emb.cpu().detach().numpy().reshape(1, -1)
    distances, indices = index.search(q_cpu, k=1)
    if distances[0][0] > threshold:
        return None
    return chunks[indices[0][0]]

def build_and_store_index(json_path, pdf_path):
    """Create the FAISS index from combined text."""
    combined = combine_sources(json_path, pdf_path)
    chunks = chunk_text(combined)
    embeddings = build_embeddings(chunks)
    index = store_embeddings_in_faiss(embeddings)
    return index, chunks

def sanitize_data(text):
    """Remove phone/email from user input, if desired."""
    phone_pat = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    email_pat = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(phone_pat, '*****', text)
    text = re.sub(email_pat, '*****', text)
    return text

def build_prompt_with_context(query, index, chunks, dataset):
    """
    1) If exact question in dataset -> direct answer
    2) Else search FAISS for a relevant chunk
    3) If no relevant chunk, or chunk doesn't clearly answer -> Je ne sais pas
    """

    # 1) Exact match from dataset
    for (q, a) in dataset:
        if query.lower() == q.lower():
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
                                            "N'oubliez pas que vous aidez les étudiants, il est donc important de répondre de manière formelle et bienveillante.  "
                          "Si vous avez un lien du site web dans la reponse affiche le   "


                        )
                    },
                    {"role": "system", "content": f"Réponse directe du dataset: {a}"},
                    {"role": "user", "content": query}
                ]
            }

    # 2) Retrieve context from PDF if no exact match
    relevant_chunk = search_relevant_context(query, index, chunks, threshold=0.9)
    if relevant_chunk:
        return {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "Tu es EKO, un assistant IA expert pour l'IHEC Carthage en Tunisie. "
                "Réponds uniquement en français. Utilise le PDF uniquement pour fournir un contexte lié à l'IHEC Carthage "
                "si aucune réponse exacte n'est trouvée dans le JSON. "
                "Si aucune information pertinente n'existe ni dans le JSON ni dans le PDF, réponds uniquement : 'Je ne sais pas.' "
                "Pour les questions sur des données personnelles, réponds : 'En tant qu'IA, je ne dois pas posséder ni traiter des données personnelles.' "
                "N'oublie pas que tu aides les étudiants. Tes réponses doivent être formelles, bienveillantes et détaillées, tout en restant charmantes pour les encourager."
                                      "Si vous avez un lien du site web dans la reponse affiche le   "

            )
        },
        {
            "role": "system",
            "content": (
                f"Contexte pertinent extrait : {relevant_chunk}" 
                if relevant_chunk else "Aucun contexte pertinent trouvé dans le PDF."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question : {query}\n"
                "Si aucune réponse ne peut être trouvée dans le contexte ou les données ci-dessus, réponds uniquement : 'Je ne sais pas.'"
            )
        }
    ]
}


    # 3) If no relevant chunk found, fallback to "Je ne sais pas."
    return {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "Tu es EKO, un assistant IA expert pour l'IHEC Carthage en Tunisie. "
                "Réponds uniquement en français. Utilise les réponses exactes du fichier JSON si elles sont disponibles. "
                "Si le JSON ne contient pas la réponse, utilise le PDF uniquement pour le contexte sur l'IHEC Carthage. "
                "Ne mentionne jamais tes sources textuellement dans tes réponses. "
                "Si aucune information n'est disponible dans le JSON ou le PDF, réponds uniquement : 'Je ne sais pas.' "
                "Pour les questions sur des données personnelles : 'En tant qu'IA, je ne dois pas posséder ni traiter de données personnelles.' "
                "N'oublie pas que tu aides les étudiants, donc il est important de répondre de manière formelle, bienveillante et claire. "
                "Ne réponds pas à des questions qui ne concernent pas l'historique ou les activités éducatives de l'IHEC Carthage."
            )
        },
        # Réponse système pour indiquer l'usage du dataset
        {
            "role": "system",
            "content": (
                f"Réponse directe depuis le dataset : {a}" 
                if a else "Aucune correspondance trouvée dans le dataset ou le PDF."
            )
        },
        # Question posée par l'utilisateur
        {
            "role": "user",
            "content": query
        }
    ]
}


# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__)

print("Building FAISS index, loading dataset, please wait...")
faiss_index, text_chunks = build_and_store_index(JSON_PATH, PDF_PATH)
dataset = load_dataset(JSON_PATH)
print("Index ready!")



@app.route("/")
def home():
    """Serve the chat page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Handles AJAX queries {query: "..."} => returns AI response."""
    data = request.get_json(force=True)
    user_query = data.get("query", "")
    if not user_query.strip():
        return jsonify({"response": "Vous n'avez rien demandé."})

    cleaned = sanitize_data(user_query)
    prompt  = build_prompt_with_context(cleaned, faiss_index, text_chunks, dataset)

    try:
        response = ollama.chat(**prompt)
        # Depending on Ollama's return format; assume it returns a dict with "message" -> "content"
        ai_reply = response["message"]["content"]
    except Exception as e:
        print("Error calling ollama:", e)
        ai_reply = "Erreur: impossible de communiquer avec le modèle."

    return jsonify({"response": ai_reply})


if __name__ == "__main__":
    app.run(debug=True)
