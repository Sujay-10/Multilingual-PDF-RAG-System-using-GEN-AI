import os
import faiss
import torch
import pytesseract
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MODEL_NAME_EMBEDDINGS = "bert-base-multilingual-uncased"  # Example multilingual model
INDEX_FILE_PATH = 'C:/Users/SHREE/Desktop/Multilingual Project/faiss_index.index'  # Updated path
EMBEDDINGS_FILE_PATH = 'C:/Users/SHREE/Desktop/Multilingual Project/embeddings.npy'  # Path to save embeddings
DATA_PATH = 'C:/Users/SHREE/Desktop/Multilingual Project/pdfs'  # Path to your PDFs
BATCH_SIZE = 5  # Define a batch size for processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chat Memory for context retention
chat_memory = deque(maxlen=5)  # Store last 5 interactions

# Load model and tokenizer
model = AutoModel.from_pretrained(MODEL_NAME_EMBEDDINGS).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_EMBEDDINGS)

def load_document(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def ocr_process(pdf_path):
    # Use OCR to extract text from scanned PDFs
    text = pytesseract.image_to_string(pdf_path)
    return text

def optimized_chunk_document(documents):
    # Use an optimized chunking algorithm (e.g., using regex for paragraph-based splitting)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=True  # Use regex to define split points
    )
    return text_splitter.split_documents(documents)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

cot_prompt_template = """
You are an advanced AI System designed to provide concise and informative answers based on the content provided.

Chain of Thought:
1. Carefully read and analyze all the provided content.
2. Understand the question being asked and identify key elements.
3. Provide a clear, accurate, and relevant answer to the question.
4. Ensure the answer is detailed yet concise, with a length of 80-100 words.
5. If the context does not contain sufficient information to answer the question, respond with 'There is no information on the asked statement'.

Provided Text:
{retrieved_text}

Provided Question:
{user_question}
"""

cot_prompt = PromptTemplate(
    input_variables=["retrieved_text", "user_question"],
    template=cot_prompt_template,
)

ollama_llm = Ollama(model='qwen2.5:1.5b')  # Ensure this model supports multilingual capabilities

def query_decomposition(user_query):
    # Example of simple query decomposition (more complex methods can be implemented)
    return user_query.split(" and ")

def hybrid_search(query, index_file_path=INDEX_FILE_PATH):
    # Load the FAISS index
    faiss_index = faiss.read_index(index_file_path)

    # Get query embedding
    query_embedding = get_embeddings([query])

    # Perform semantic search
    distances, indices = faiss_index.search(query_embedding, k=5)  # Retrieve top 5 relevant documents
    keyword_results = []  # Implement keyword search logic here

    # Combine keyword and semantic search results
    combined_indices = list(set(indices.flatten().tolist() + keyword_results))  # Deduplicate results
    return combined_indices

def rerank_results(retrieved_chunks, user_question):
    # Simple reranking based on some heuristic or scoring
    # This can be expanded with more sophisticated models
    return sorted(retrieved_chunks, key=lambda x: len(x))[:5]  # Example: rank by length of content

def metadata_filtering(retrieved_chunks, metadata):
    # Filter retrieved chunks based on some metadata criteria
    return [chunk for chunk in retrieved_chunks if chunk.metadata['type'] in metadata]  # Example filter

def search_chat_memory(user_query, memory):
    """
    Search through chat memory to find any relevant question or answer
    based on the user's query.
    """
    # Convert the user query and memory to lowercase for case-insensitive search
    user_query_lower = user_query.lower()

    # Iterate through chat memory and find relevant responses
    matches = []
    for q, a in memory:
        if user_query_lower in q.lower() or user_query_lower in a.lower():
            matches.append(f"Q: {q}\nA: {a}")

    if matches:
        return "\n\n".join(matches)  # Return all matched conversations
    else:
        return "No relevant conversation found in the memory."

def handle_meta_queries(user_query, memory):
    """
    Identify and handle meta-level queries such as asking about previous conversations.
    """
    # Meta-query keywords: looking for "previous", "conversation", "before", etc.
    if "previous" in user_query.lower() or "conversation" in user_query.lower() or "before" in user_query.lower():
        # Search the memory for any related past questions/answers
        return search_chat_memory(user_query, memory)
    return None  # No meta-query detected

def generate_final_answer(retrieved_chunks, user_question):
    # First check for meta-level queries
    meta_response = handle_meta_queries(user_question, chat_memory)
    if meta_response:
        return meta_response  # Return the response from chat memory

    combined_text = "\n\n".join(retrieved_chunks)  # Combine all relevant chunks

    # Create a context string from chat memory
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_memory])
    
    # Format the prompt for summarization or question answering
    formatted_prompt = cot_prompt.format_prompt(retrieved_text=combined_text, user_question=user_question)
    
    # Include memory context in the prompt
    if context:
        prompt_text = f"{context}\n\n{formatted_prompt.to_string()}"
    else:
        prompt_text = formatted_prompt.to_string()

    # Get the final answer from the language model
    final_answer = ollama_llm(prompt_text)
    
    # Store the current question and answer in memory
    chat_memory.append((user_question, final_answer))

    return final_answer

def process_documents_in_batches(data_path, batch_size):
    all_chunks = []
    documents = load_document(data_path)

    # Load and process documents in batches
    for i in range(0, len(documents), batch_size):
        batch_documents = documents[i:i + batch_size]
        batch_chunks = optimized_chunk_document(batch_documents)
        all_chunks.extend(batch_chunks)

    return all_chunks

import time
if __name__ == '__main__':
    # Load documents initially in batches
    chunks = process_documents_in_batches(DATA_PATH, BATCH_SIZE)

    # Load or compute embeddings and create the FAISS index
    if os.path.exists(EMBEDDINGS_FILE_PATH):
        print("Loading existing embeddings...")
        embeddings = np.load(EMBEDDINGS_FILE_PATH)

        if os.path.exists(INDEX_FILE_PATH):
            print("Loading existing FAISS index...")
            faiss_index = faiss.read_index(INDEX_FILE_PATH)
        else:
            print("Creating FAISS index...")
            faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            faiss_index.add(embeddings)
            faiss.write_index(faiss_index, INDEX_FILE_PATH)
    else:
        print("Computing and saving embeddings...")
        embeddings = get_embeddings([chunk.page_content for chunk in chunks])
        np.save(EMBEDDINGS_FILE_PATH, embeddings)
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, INDEX_FILE_PATH)

    first_query = True
    while True:
        if first_query:
            user_query = input("What is your Question: ")
            first_query = False  # Update the flag after the first query
        else:
            user_query = input("Do you want to know anything more? (type Q to quit): ")

        if user_query.lower() == 'q':
            print("Exiting the program.")
            break

        start_time = time.time()

        # Decompose the query if necessary
        decomposed_queries = query_decomposition(user_query)

        # Combine results from hybrid search
        combined_indices = []
        for query in decomposed_queries:
            relevant_indices = hybrid_search(query, INDEX_FILE_PATH)
            combined_indices.extend(relevant_indices)

        # Filter out indices that are out of bounds
        valid_indices = [i for i in set(combined_indices) if i < len(chunks)]
        if not valid_indices:
            print("No relevant chunks found for the query.")
            continue

        # Retrieve relevant chunks based on combined indices
        retrieved_chunks = [chunks[i].page_content for i in valid_indices]

        # Rerank results for relevance
        reranked_chunks = rerank_results(retrieved_chunks, user_query)

        # Generate a single final answer based on all relevant chunks
        final_answer = generate_final_answer(reranked_chunks, user_query)

        end_time = time.time()
        time_taken = end_time - start_time

        # Display the final answer
        print(f"The Answer that you are looking for is:\n{final_answer}\n")
        print('Time taken to generate the answer is {} seconds'.format(time_taken))
