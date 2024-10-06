# Multilingual-PDF-RAG-System-using-GEN-AI  

Overview:-  
This project implements a Retrieval-Augmented Generation (RAG) system designed to process multilingual PDFs, extracting information and providing summaries and answers based on their content. The system is capable of handling PDFs in various languages, including Hindi, English, Bengali, and Chinese, and processes both scanned and digital documents.  

##Features:-  
1.**Text Extraction**: Utilizes Optical Character Recognition (OCR) for scanned documents and standard text extraction methods for digital PDFs, ensuring comprehensive data processing.  
2.**Chunking Algorithms**: Implements optimized chunking techniques to divide documents into manageable sections, facilitating effective information retrieval.  
3.**Chat Memory**: Maintains a memory of recent interactions to provide contextually relevant responses to user queries, enhancing the conversational experience.  
4.**Query Decomposition**: Breaks down complex user queries into simpler components, improving the accuracy of information retrieval.  
5.**Hybrid Search**: Combines keyword-based and semantic search techniques, utilizing a FAISS index for efficient vector-based searching.  
6.**Reranking**: Enhances the relevance of search results through a reranking algorithm, ensuring the most pertinent information is prioritized.  
7.**Integration with High-Performance Vector Databases**: Capable of scaling to handle large datasets (up to 1TB), making it suitable for extensive document collections.  

##Technologies Used:-  
1.**Python**: The primary programming language for implementation.  
2.**PyTorch & Transformers**: For model handling and text embeddings using BERT-based multilingual models.  
3.**Langchain**: A framework that simplifies interactions with language models and text processing.  
4.**FAISS**: A library for efficient similarity search and clustering of dense vectors.  
5.**Pytesseract**: An OCR tool for text extraction from images.  
6.**LLM Model**: Used the Llama3.2 LLM model using the Ollama to run the model locally.  

##Getting Started:-  
1.Clone this repository.  
2.Install the required dependencies using 'pip install -r requirements.txt'.  
3.Place your PDF documents in the pdfs directory.  
4.Run the script and interact with the system by asking questions related to the content of your PDFs.  
5.Run the command 'ollama pull llama3.2:1b' and 'ollama pull qwen2.5:1.5b' in the terminal.  

(If the installing of dependancies doesn't work, install the dependencies manually using the below command:
  'pip install langchain torch transformers ollama numpy deque faiss')
