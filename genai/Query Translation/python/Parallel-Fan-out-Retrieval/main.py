from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain.schema.runnable import RunnableParallel, RunnableLambda

# 1. SETUP AND MODEL INITIALIZATION
# Load environment variables (e.g., API keys) from a .env file
load_dotenv()

# Initialize LLM-1 and LLM-2 for query generation and final response
# (Both use the same model in this example) [cite: 29, 44]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

# Initialize the embedding model for converting text to vectors
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Define the expected structured output for the generated queries.
# This ensures LLM-1 returns exactly three queries. [cite: 76, 77]
class Queries(BaseModel):
    res1:str = Field(description='write down 1st query from user input')
    res2:str = Field(description='write down 2nd query from user input')
    res3:str = Field(description='write down 3nd query from user input')

# Apply the Pydantic structure to the LLM for structured output
llm_strutured = llm.with_structured_output(Queries)

# 2. USER QUERY INPUT
query = 'what is js'
print("query ->", query)
print("-------------------------------------------------------------")

# 3. QUERY TRANSFORMATION (GENERATION - LLM-1)
system = """
from the given user query generates 3 simillar types of queries.
    for e.g
    input:
    query - how microsoft start
    output:
    res1 - who funded microsoft and when
    res2 - history of microsoft origin
    res3 - begeinning phase of microsoft
""" # System prompt to guide LLM-1 to generate three diverse queries [cite: 88, 89, 90, 91, 92, 93, 94, 95]

# Create the prompt for LLM-1 [cite: 97, 99, 100]
chat_prompt = ChatPromptTemplate([
    ("system","{system}"),
    ("human","user query - {query}")
])

prompt = chat_prompt.invoke({"system":system,"query":query})

# Invoke LLM-1 to generate the three new queries [cite: 49]
res = llm_strutured.invoke(prompt)

print("genrated queries ✅")
print(res)
print("-------------------------------------------------------------")

# 4. PARALLEL RETRIEVAL SETUP
# Initialize three separate retriever objects pointing to the same Vector DB collection.
# This is a common pattern for parallel search in LangChain. [cite: 51, 107, 115, 119]
reterival1 = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="js_notes",
    embedding=embedder
)
reterival2 = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="js_notes",
    embedding=embedder
)
reterival3 = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="js_notes",
    embedding=embedder
)

# Define a parallel runnable chain. This executes all three retrieval calls concurrently.
parallel_reterival = RunnableParallel({
    "reterival1": RunnableLambda(lambda _: reterival1.similarity_search(query=res.res1)), # Retrieve for Query 1
    "reterival2": RunnableLambda(lambda _: reterival2.similarity_search(query=res.res2)), # Retrieve for Query 2
    "reterival3": RunnableLambda(lambda _: reterival3.similarity_search(query=res.res3)), # Retrieve for Query 3
})

print("parallel query reterival ✅")
print("-------------------------------------------------------------")

# Execute the parallel retrieval
data = parallel_reterival.invoke(None)

# 5. CONTEXT MERGING (Filtering Unique Documents)
# Combine all retrieved documents into one list [cite: 54, 135]
combined_list = data['reterival1'] + data['reterival2'] + data['reterival3']

# Filter out duplicate documents to retain only unique context.
# Duplicates are checked based on a document's 'page_label' metadata. [cite: 54, 136]
unique_docs = []
seen_labels = set()

for doc in combined_list:
    label = doc.metadata.get('page_label')
    # Check if the label exists and hasn't been seen before
    if label and label not in seen_labels:
        seen_labels.add(label)
        unique_docs.append(doc)

# Concatenate the content of the unique documents to form the final context
unique_page_contents = [doc.page_content for doc in unique_docs]
combined_answer = "\n\n".join(unique_page_contents)

print("combined_query reterival (Unique Documents Filtered)")
print("-------------------------------------------------------------")
# print(combined_answer) # Uncomment to see the retrieved context

# 6. ANSWER GENERATION (LLM-2)
# Define the final prompt for the Answer Generation phase [cite: 55, 146]
chat_prompt = ChatPromptTemplate([
    # System prompt: Instruct the LLM to act as an assistant and ground its answer 
    # ONLY in the provided combined context (combined_answer). [cite: 148]
    ('system', 'You are an assistant helping to answer user queries based on the following content. If the answer to the query is not found in the content, respond with: "This query is not present in the document." Here is the content:\n\n{combined_answer}'),
    # Human prompt: Pass the original user query to LLM-2. [cite: 59, 150]
    ('human', 'Explain in simple terms: what is {query}?')
])

prompt = chat_prompt.invoke({"combined_answer":combined_answer,"query":query})

# Invoke LLM-2 to generate the final, grounded response [cite: 60]
res = llm.invoke(prompt)

print("Answer--->",res.content)