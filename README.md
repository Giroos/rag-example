# General

- An example of RAG application.
- Able to answer questions about the kaggle FAQ, original dataset: https://www.kaggle.com/datasets/ikjotsingh221/kagglefaq?select=kaggle_learn_faq.txt

# Requirements to execute the application:
- Epam dial api key
- You need to be connected to Epam VPN
- Python 3.12 or Docker

# How to run using docker:
- if you want to rebuild the vector embeddings, delete the `embeddings_index.pkl` file
- inside the project folder, create `.env` file, with the content `DIAL_API_KEY=<your dial api key>`
- $ `docker build -t gj-rag-hometask .`
- $ `docker run --rm -it --env-file .env gj-rag-hometask`

# How to run using python:
- if you want to rebuild the vector embeddings, delete the `embeddings_index.pkl` file
- create an environment variable `DIAL_API_KEY=<your dial api key>`
- install requirements: $ `pip install -r requirements.txt`
- run the `app.py` file

# Usage explanation:
- you can ask questions only about kaggle. The system should reject other questions.
- The application's answer contains 3 sections:
  - Similarity search query - the query used to search for relevant information
  - Answer - the answer to the user's question
  - References - the referred information sources
- The application does not use memory, each question is a blank conversation

# Technical decisions explanation:
- Data files parsing and chunking (see `utils.py`)
  - The data source contains txt and pdf files, the pdf files are first converted to text using `pypdf` library
  - Then the text is splitted by paragraph (double line break), very short chunks are discarded. This is simple but effective method for this particular use case.
  - the source file name and page number of each chunk is appended to metadata, to provide the references in the answer to user's question
- Embeddings
  - The embeddings are created using the `text-embedding-3-small-1` model 
  - The embeddings are stored in memory, because the dataset is relatively small
- Prompt engineering and retrieval:
  - The application first creates a search query using LLM, to eliminate possible noise and translate to english, if necessary
  - using the created search query, vector search is performed
    - using cosine similarity. OpenAI embeddings are directional representations of meaning, cosine similarity measures the angle between 2 vectors, ignoring the magnitude (length) of the vector, which can vary for many reasons that are not directly related to the meaning.
    - top_k = 5 chosen experimentally, observed to provide the best results
  - the LLm is asked the original user's question and provided with the retrieved data as a context.
  - The LLM is instructed to decline questions that are not related to kaggle, or to the information in the datasource. Prevents not intended usage of the application.
    - Done in the second call to LLM, because we don't have the retrieved context in the first call.
  - The output is limited to 500 tokens, for cost control and better user experience.
