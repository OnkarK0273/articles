from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
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
query_genration = (prompt_decomposition | llm | StrOutputParser() | (lambda x : x.split("\n")))

# --- Reterived sub-quetion ans generation prompt ---
reterived_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""
reterived_prompt = ChatPromptTemplate.from_template(reterived_template)

# --- Utility:contenxt merging subquetion with answers ---
def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

# --- each subquetions run in concurrently -> reterived document -> and gen. thier reponse 

def retrieve_and_rag(question,query_genration,reterived_prompt):
    """RAG on each sub-question"""
    sub_questions = query_genration.invoke({"question":question})

    q_a_pairs = ""
    for sub_question in sub_questions:
        
        chain = (
            {
            "question":itemgetter("question") ,
            "q_a_pairs":itemgetter("q_a_pairs"),
            "context":itemgetter("question") | reteriver
            }|
            reterived_prompt|
            llm|
            StrOutputParser()
        )

        ans = chain.invoke({"question":sub_question,"q_a_pairs":""})
        q_a_pair = format_qa_pair(sub_question,ans)
        q_a_pairs = q_a_pairs + "\n-----\n" + q_a_pair
    
    return q_a_pairs

question = 'what is js'
context = retrieve_and_rag(question,query_genration,reterived_prompt)

# --- Prompt for final answer ---
template = """Here is a set of Q+A pairs:
{context}
Use these to synthesize an answer to the question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- Final RAG chain ---
final_chain = (prompt | llm | StrOutputParser() )

# --- Run the chain ---
if __name__ == "__main__":
    final_ans = final_chain.invoke({"context":context,"question":question})
    print("Final answer:", final_ans)