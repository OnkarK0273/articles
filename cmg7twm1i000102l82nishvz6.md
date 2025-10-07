---
title: "Query Translation (Query Re-writing) [Code Explanation]"
seoTitle: "Query Translation Code Explained"
seoDescription: "Enhance RAG pipelines with advanced query transformation using Multi-Query, RAG Fusion, and HyDE in LangChain"
datePublished: Wed Oct 01 2025 10:13:41 GMT+0000 (Coordinated Universal Time)
cuid: cmg7twm1i000102l82nishvz6
slug: query-translation-query-re-writing-code-explanation
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1759313487010/316f2068-ad23-4f77-a862-983a9cabafb1.png
tags: query-rewrite, query-transformation, query-translation

---

This articles demonstrates three advanced Query Transformation techniques—Multi-Query, RAG Fusion, and HyDE—used to improve document retrieval in a RAG pipeline using LangChain and Google Generative AI.

if u don’t know about Query Re-writing visit our [articles](https://onkark.hashnode.dev/query-translation-query-re-writing-advance-rag), here u learn what is Query Re-writing and their methods

# Setup and Initialization

1. ## Setup `.env` file
    

```plaintext
# gemini api key - https://aistudio.google.com/api-keys
GOOGLE_API_KEY = < api key > 
```

2. ## Setup `main.py` file
    

First, we set up the environment, load models, and configure the connection to the vector store (Qdrant) and the embedding model.

```python
# Install necessary libraries (if not already installed)
# !pip install -q -U langchain langchain-google-genai python-dotenv qdrant-client pydantic

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
# --- Load environment variables ---
load_dotenv()

# --- LLM and Embedding setup ---
# LLM for final answer synthesis (and query generation/HyDE if applicable)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
# Embedding model for converting text to vector space
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# --- Vector store and retriever setup ---
# Connect to the existing Qdrant collection
vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="js_notes",
    embedding=embedder
)
# Create a basic retriever instance
retriever = vector_store.as_retriever()
```

# 1\. Multi-Query Retrieval (Parallel Retrieval)

here is complete [GitHub code](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Parallel-Fan-out-Retrieval/main.py) of Multi-Query Retrieval

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759303772300/ca2af3f8-5ad6-4572-90e5-0525304989ee.png align="center")

## 1.1. Query Generation and De-Duplication Logic

```python
# --- Multi-query generation prompt ---
multi_query_template = """
You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries (3 queries) related to: {question}
Output (only generate queries):
query1
query2
query3
"""
multi_query_prompt = ChatPromptTemplate.from_template(multi_query_template)

# --- Query Generation Chain (LLM-1) ---
generate_queries_chain = (
    multi_query_prompt
    | llm
    | StrOutputParser()
    | (lambda output: output.split("\n")) # Transform LLM string output into a list of queries
)

# --- Utility: Unique union of retrieved documents (Context Merging) ---
def get_unique_union(documents: List[List[Any]]):
    """
    Flattens a list of lists of documents and removes duplicates based on serialization.
    This is the core context merging step for Multi-Query.
    """
    # 1. Flatten the list of results from all parallel retrievals
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 2. Use a set to efficiently find unique documents
    unique_docs = list(set(flattened_docs))
    # 3. Deserialize back into LangChain Document objects
    return [loads(doc) for doc in unique_docs]

# --- Retrieval Chain ---
retrieval_chain_multi_query = (
    generate_queries_chain
    | retriever.map() # Executes the retriever for each generated query in parallel
    | get_unique_union # Merges and de-duplicates all results
)
```

## 1.2. Final Execution

```python
# --- Final Multi-Query RAG chain: retrieve context, format prompt, get answer from LLM ---
answer_template = """Answer the following question based on this context:
{context}
Question: {question}
"""
answer_prompt = ChatPromptTemplate.from_template(answer_template)

final_rag_chain_multi_query = (
    {
        "context": retrieval_chain_multi_query,
        "question": itemgetter("question")
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# user_question = "what is js"
# answer = final_rag_chain_multi_query.invoke({"question": user_question})
# print("Multi-Query Final answer:", answer)
```

# 2\. RAG Fusion (Multi-Query + RRF)

here is complete [GitHub code](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Reciprocal-Rank-Fusion/main.py) of RAG Fusion

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759310073708/55678efc-bc0c-4df3-b031-b6246653404e.png align="center")

## 2.1. RRF Logic

We define the RRF function, which replaces the simple de-duplication step. It calculates a fused score for every document across all retrieved lists.

```python
# --- Reciprocal Rank Fusion Function (Context Merging with Re-ranking) ---
def reciprocal_rank_fusion(results: List[List[Any]], k=60):
    """
    Combines multiple lists of ranked documents using Reciprocal Rank Fusion (RRF) formula:
    RRF(d) = Σ(i=1 to n) [ 1 / (rank_i(d) + k) ]
    """
    fused_scores = {}
    for docs in results: # Iterate through each retrieved list
        for rank, doc in enumerate(docs): # rank is 0-indexed
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            # Apply RRF formula: 1 / (rank + k)
            # Use rank + 1 since enumerate is 0-indexed, and k is the constant (typically 60)
            fused_scores[doc_str] += 1 / (rank + 1 + k) 
            
    # 1. Sort documents by their calculated RRF score (descending)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # 2. Return only the re-ranked documents (top N can be selected here to manage context size)
    return [doc for doc, score in reranked_results] 

# --- Retrieval Chain (Parallel Retrieval + RRF) ---
# Note: The 'generate_queries_chain' from the Multi-Query section is reused here.
retrieval_chain_rrf = (
    generate_queries_chain
    | retriever.map() # Parallel retrieval
    | reciprocal_rank_fusion # Merges, scores, and re-ranks all results
)
```

### 2.2. Final Execution

```python
# --- Final RAG Fusion RAG chain: retrieve context, format prompt, get answer from LLM ---
final_rag_chain_rrf = (
    {
        "context": retrieval_chain_rrf,
        "question": itemgetter("question")
    }
    | answer_prompt # Reuses the same answer prompt
    | llm
    | StrOutputParser()
)

# user_question = "what is javascript"
# answer = final_rag_chain_rrf.invoke({"question": user_question})
# print("RAG Fusion Final answer:", answer)
```

# 3.Hypothetical Document Embeddings (HyDE)

here is complete [GitHub code](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/HyDE/main.py) of Hypothetical Document Embeddings (HyDE)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759310451366/d0b6e31e-794a-4cd4-ba22-1b9574a83339.png align="center")

## 3.1. HyDE Generation and Retrieval Logic

We define a specific prompt to generate a detailed hypothetical document, and use a potentially larger LLM for this generation step for better detail.

```python
# --- HyDE generation prompt (LLM-1) ---
template_hyde = """
Please write a scientific paper passage to answer the question
Question: {question}
Passage
"""
prompt_hyde = ChatPromptTemplate.from_template(template_hyde)

# Use a potentially larger LLM for better hypothetical document quality
large_parameter_llm = ChatGoogleGenerativeAI(model ='gemini-2.5-flash')

# --- Hypothetical Document Generation Chain ---
generate_docs_for_retrieval = ( 
    prompt_hyde 
    | large_parameter_llm 
    | StrOutputParser() # Outputs the hypothetical document text
)

# --- Retrieval Chain (HyDE document -> Embedding -> Retrieval) ---
retrieval_chain_hyde = (
    generate_docs_for_retrieval # The output text of the HyDE document
    | retriever # LangChain embeds this text automatically and searches the VectorDB
)
```

## 3.2. Final Execution

```python
# --- Final HyDE RAG chain: retrieve context, format prompt, get answer from LLM ---
final_rag_chain_hyde = (
    {
        "context":retrieval_chain_hyde, # Real documents retrieved based on HyDE embedding
        "question":itemgetter("question") 
    } 
    | answer_prompt # Reuses the same answer prompt
    | llm 
    | StrOutputParser()
)

# user_question = "what is js"
# answer = final_rag_chain_hyde.invoke({"question": user_question})
# print("HyDE ", answer)
```

# Resources

## 1.Github code

* [Multi-Query Retrieval](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Parallel-Fan-out-Retrieval/main.py)
    
* [RAG Fusion](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Reciprocal-Rank-Fusion/main.py)
    
* [Hypothetical Document Embeddings](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/HyDE/main.py)
    

## 2.Articles

* [Multi-Query Retrieval](https://onkark.hashnode.dev/query-translation-query-re-writing-advance-rag#heading-1-parallel-fan-out-retrieval)
    
* [RAG Fusion](https://onkark.hashnode.dev/query-translation-query-re-writing-advance-rag#heading-2-reciprocal-rank-fusion-rrf)
    
* [Hypothetical Document Embeddings](https://onkark.hashnode.dev/query-translation-query-re-writing-advance-rag#heading-3-hypothetical-document-embeddings-hyde)