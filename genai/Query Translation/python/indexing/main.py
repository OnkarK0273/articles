from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# embeddding

embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# loaders

loader = PyPDFLoader('./pdf/nodejs.pdf')
doc = loader.load()

# splitting 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_doc = text_splitter.split_documents(doc)

# vectore

vectore = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="nodejs",
    embedding=embedder
)

res = vectore.add_documents(split_doc)

print("res->",res)


