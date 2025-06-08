import os
import pickle
import glob
import numpy as np
from tqdm import tqdm
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_pdf, load_txt

# --- AzureOpenAI setup ---
KEY = os.environ["DIAL_API_KEY"]

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key=KEY,
)

embedding_deployment = 'text-embedding-3-small-1'
llm_deployment = 'gpt-4o-mini-2024-07-18'

# --- Build or load embeddings index ---

def build_index(data_folder='data'):
    all_chunks = []
    for file_path in glob.glob(os.path.join(data_folder, '*')):
        filename = os.path.basename(file_path)
        if filename.lower().endswith('.pdf'):
            chunks = load_pdf(file_path)
        elif filename.lower().endswith('.txt'):
            chunks = load_txt(file_path)
        else:
            continue
        for c in chunks:
            c["source"] = filename
        all_chunks.extend(chunks)

    print(f"Loaded {len(all_chunks)} chunks from documents.")

    texts = [c["text"] for c in all_chunks]

    batch_size = 32
    embeddings = []

    print("Computing embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=embedding_deployment,
            input=batch
        )
        batch_embeds = [np.array(d.embedding) for d in response.data]
        embeddings.extend(batch_embeds)

    index = {
        "chunks": all_chunks,
        "embeddings": np.vstack(embeddings)
    }

    with open('embeddings_index.pkl', 'wb') as f:
        pickle.dump(index, f)

    print("Index saved to embeddings_index.pkl")
    return index

def load_index():
    with open('embeddings_index.pkl', 'rb') as f:
        return pickle.load(f)

# --- Retrieval ---

def retrieve(query, index, top_k=5):
    response = client.embeddings.create(
        model=embedding_deployment,
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, index['embeddings'])[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        chunk = index['chunks'][idx]
        score = similarities[idx]
        retrieved.append((chunk, score))

    return retrieved

# --- Answer generation ---


def create_query(user_prompt):
    prompt = f"""
    You are an automated QA system. You have a knowledge base about kaggle. You are given the user prompt and you need to create a query text for similarity search in your knowledge base, based on the user prompt.
    The query text should always be in english, even if the user asks in different language.

    User prompt: {user_prompt}

    query text:"""

    response = client.chat.completions.create(
        model=llm_deployment,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return answer


def answer_question(user_question, index, query):
    retrieved_chunks = retrieve(query, index)
    context = ""
    references = []
    for c, score in retrieved_chunks:
        context += f"\n---\nSource: {c['source']} - Page {c['page']}\n{c['text']}\n"
        references.append(f"{c['source']} - Page {c['page']}")

    prompt = f"""
    You are given with the information from kaggle FAQ. You should answer questions only about kaggle. You should answer the question based only on the provided context. 
    If the answer is not in the context, or if the question is not about kaggle, say "I don't have the answer.".

    Question: {user_question}

    Context: {context}

    Answer:"""

    response = client.chat.completions.create(
        model=llm_deployment,
        messages=[
            {"role": "system", "content": "You need to provide short answers to the users questions related to kaggle. If the user question is not related to kaggle, say 'I don't have the answer.'"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return answer, references


if __name__ == "__main__":
    # Build index only if embeddings_index.pkl doesn't exist
    if not os.path.exists('embeddings_index.pkl'):
        build_index()

    index = load_index()

    while True:
        q = input("\nEnter your question (or 'exit'): ")
        if q.lower() == 'exit':
            break
        query = create_query(q)
        print("\nSimilarity search query:\n", query)
        answer, refs = answer_question(q, index, query)
        print("\nAnswer:\n", answer)
        print("\nReferences:")
        for ref in refs:
            print("-", ref)