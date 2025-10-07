from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from operator import itemgetter
load_dotenv()


# --- LLM and Embedding setup ---
llm = ChatGroq(model="llama-3.1-8b-instant")
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# --- Vector store setup ---
vectore = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="js_notes",
    embedding= embedder
)

reteriver = vectore.as_retriever()


# --- Query-decomposition generation prompt ---
template_decomposition = """
You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
note: generate only queries
Output (only 3 queries): 
"""

prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)

# --- Query Generation Chain ---
query_genration = ( prompt_decomposition | llm | StrOutputParser() | (lambda x : x.split("\n") ) ) 

# --- Reterived sub-quetion ans generation prompt ---
reterived_template ="""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n
Question: {question} \n
Context: {context} \n
Answer:
"""
reterived_prompt = ChatPromptTemplate.from_template(reterived_template)

# --- Utility:contenxt merging subquetion with answers ---
def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

# --- Utility: Sub-quetions and thier answers ---
def retrieve_and_rag(question,query_genration,reterived_prompt):
    """RAG on each sub-question"""
    sub_questions = query_genration.invoke({"question":question})

    rag_results = []

    for sub_question in sub_questions:

        chain = (
            {
                "question":itemgetter("question"),
                "context":itemgetter("question") | reteriver
            }|
            reterived_prompt |
            llm |
            StrOutputParser()
        )

        ans = chain.invoke({"question":sub_question})
        rag_results.append(ans)
    
    
    context = format_qa_pairs(sub_questions, rag_results)
    return context

question = 'what is js'
context  = retrieve_and_rag(question,query_genration,reterived_prompt)

# --- Prompt for final answer ---
template = """Here is a set of Q+A pairs:
{context}
Use these to synthesize an answer to the question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- Final RAG chain ---
final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# --- Run the chain ---
if __name__ == "__main__":
    final_ans = final_rag_chain.invoke({"context":context,"question":question})
    print("Final answer:", final_ans)