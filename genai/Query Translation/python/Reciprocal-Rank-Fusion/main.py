from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from operator import itemgetter
from langchain.load import dumps, loads

# Load environment variables
load_dotenv()

# --- Query Generation Prompt ---
query_generation_template = """
You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries (3 queries) related to: {question}
Output (only generate queries):
query1
query2
query3
"""
query_prompt = ChatPromptTemplate.from_template(query_generation_template)

# --- LLM and Embedding setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# --- Query Generation Chain ---
generate_queries_chain = (
    query_prompt
    | llm
    | StrOutputParser()
    | (lambda output: output.split("\n"))
)

# --- Vector store and retriever setup ---
vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="js_notes",
    embedding=embedder
)
retriever = vector_store.as_retriever()

# --- Reciprocal Rank Fusion Function ---
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Combines multiple lists of ranked documents using Reciprocal Rank Fusion (RRF).
    Documents appearing in multiple lists get higher scores.
    """
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

# --- Retrieval Chain ---
retrieval_chain = (
    generate_queries_chain
    | retriever.map()
    | reciprocal_rank_fusion
)

# --- Prompt for Final Answer ---
answer_template = """Answer the following question based on this context:
{context}
Question: {question}
"""
answer_prompt = ChatPromptTemplate.from_template(answer_template)

# --- Final RAG chain: retrieve context, format prompt, get answer from LLM  ---
final_rag_chain = (
    {
        "context": retrieval_chain,
        "question": itemgetter("question")
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# --- Run the Chain ---
if __name__ == "__main__":
    user_question = "what is javascript"
    answer = final_rag_chain.invoke({"question": user_question})
    print("final-response:", answer)