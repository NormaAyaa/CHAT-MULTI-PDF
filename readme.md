# Chat with Multiple PDFs

Welcome to the Chat with Multiple PDFs GitHub repository! This project is a Streamlit web application that allows users to interact with a chatbot capable of answering questions related to multiple PDF documents. The chatbot uses AI and natural language processing to provide responses based on the content of the uploaded PDFs.


## Features
- Upload and process local PDF files.
- Enter URLs of PDFs to extract and process their content.
- Create a vector database from processed PDFs for efficient retrieval.
- Load and use pre-existing vector databases.
- Chat with the AI chatbot to ask questions related to the PDFs.
- Save the newly created vector database for future use.

## How It Works
![MultiPDF Chat App Diagram](./rag.png) [source](https://miro.medium.com/v2/resize:fit:1127/1*Jq9bEbitg1Pv4oASwEQwJg.png)

1. PDF Processing: The app imports multiple PDF documents and extracts their textual content.

2. Text Segmentation: The extracted text is divided into smaller, manageable sections for efficient handling.

3. Language Modeling: Our application employs a language model to create vector representations (embeddings) of these text segments.

4. Matching Semantics: When you pose a question, the app assesses it against the text segments and identifies those that are most semantically similar.

5. Response Generation: The selected text segments are then passed to the language model, which generates responses based on the pertinent information from the PDFs.

6. Respond Responsiblliy: If the question is out of context provided via PDFs, the output response is simply "I don't know".


## Getting Started

### Prerequisites
- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://github.com/mstamy2/PyPDF2)
- [dotenv](https://pypi.org/project/python-dotenv/)
- [langchain](https://docs.langchain.com/docs/)

### Installation
1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/PC-FSU/multi-pdf-query.git

2. Change to the project directory:

   ```bash
      cd multi-pdf-query

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
   
4. Rename `.my_env` file to `.env`. Add your open API key (Huggingface key(optional)) to `.env` file.

   
### Usage

1. Run the Streamlit app:

   ```bash
      streamlit run app.py

2. Open a web browser and navigate to the local URL provided by Streamlit (e.g., http://localhost:8501).

3. Upload PDF files or enter PDF URLs to process them.

4. Click on "Create Database" to generate a vector database from the processed PDFs.

5. Chat with the AI chatbot by entering your questions in the chat input.

6. Optionally, you can save the newly created vector database for future use by clicking "Save the newly created Database."

7. To load a pre-existing vector database, enter the path to the database file and click "Load Database."

### To-Do.
-  Add support to link with web hosted vector-database like [pinecone](https://www.pinecone.io/) etc.
