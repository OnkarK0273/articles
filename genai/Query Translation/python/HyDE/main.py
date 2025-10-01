from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# --- Load environment variables ---
load_dotenv()

# --- HyDE generation prompt ---
template = """
Please write a scientific paper passage to answer the question
Question: {question}
Passage
"""
prompt_hyde = ChatPromptTemplate.from_template(template)

# --- LLM and Embedding setup ---
large_parameter_llm = ChatGoogleGenerativeAI(model ='gemini-2.5-flash')
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# --- Hypothetical Document Generation Chain ---
generate_docs_for_retrieval = ( prompt_hyde | large_parameter_llm | StrOutputParser() )

# --- Vector store and retriever setup ---
vectore = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="js_notes",
    embedding=embedder
)
reteriver = vectore.as_retriever()

# --- reteriver doc Chain ---
retrieval_chain = (generate_docs_for_retrieval | reteriver )

# --- Prompt for final answer ---
template2 = """Answer the following question based on this context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template2)

# --- Final RAG chain:
final_rag_chain = ({"context":retrieval_chain ,"question":itemgetter("question") } | prompt | llm | StrOutputParser())

# --- Run the chain ---
if __name__ == "__main__":
    user_question = "what is js"
    answer = final_rag_chain.invoke({"question": user_question})
    print("Final answer:", answer)