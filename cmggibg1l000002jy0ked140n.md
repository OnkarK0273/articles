---
title: "Query Translation (Query Decomposition) [Code Explanation]"
seoTitle: "Query Decomposition Code Explained"
seoDescription: "Explore advanced query decomposition techniques using LangChain and Google Generative AI to enhance document retrieval in RAG pipelines"
datePublished: Tue Oct 07 2025 11:59:13 GMT+0000 (Coordinated Universal Time)
cuid: cmggibg1l000002jy0ked140n
slug: query-translation-query-decomposition-code-explanation
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1759835986481/700c4d06-ceff-4cda-a33b-451dd6bbe099.png
tags: query-transformation, query-decomposition, query-translation

---

This articles demonstrates two advanced Query Decomposition techniques—Parallel and Iterative decomposition—used to improve document retrieval in a RAG pipeline using LangChain and Google Generative AI.

if u don’t know about Query Decomposition visit our [**articles**](https://onkark.hashnode.dev/query-decomposition), here u learn what is Query Decomposition and their methods.

# Common Setup

here are common things are required for both methods.

## 1\. `.env` setup

```python
# gemini api key - <https://aistudio.google.com/api-keys>
GOOGLE_API_KEY = < api key >
```

## 2\. **Imports setup**

here are the imports required for the code

```python
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
# --- Load environment variables ---
load_dotenv()
```

## 3\. **LLM and Embedding Setup**

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">Here i using google’s Gemini model for text and embedding generation, you can use any other of your choice.</div>
</div>

* `ChatGoogleGenerativeAI` for text generation
    
* `GoogleGenerativeAIEmbeddings` for vector embedding
    

```python
# --- LLM and Embedding setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Embedding model for converting text to vector space
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"
```

## 4\. **Vector Store and Retriever**

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">Here i using Qdrant vectorDB, you can use any other of your choice.</div>
</div>

* `QdrantVectorStore` for document retrieval.
    
* for semantic search we use `as_retriever`
    

```python
# --- Vector store setup ---
# vectoreDB config.
vectore = QdrantVectorStore.from_existing_collection(
    url="<http://localhost:6333>",
    collection_name="js_notes",
    embedding= embedder
)
# semantic serch to reterive document from vectoreDB
reteriver = vectore.as_retriever()
```

# **1\. Parallel** Query **Decomposition**

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">GitHub code file - <a target="_self" rel="noopener noreferrer" class="notion-link-token notion-focusable-token notion-enable-hover" href="https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Parallel-Decomposition/main.py" style="pointer-events: none">link</a></div>
</div>

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759836383448/4ad45a77-f62f-484a-9184-b195233b9d18.png align="center")

## 1\. **Prompt**

* `template_decomposition` - a prompt instruction that tells llm to break down complex questions into 3 sub-questions
    
* `prompt_decomposition` - `template_decomposition` it wraps around `prompt_decomposition` to use it in chain
    

```python
# --- Query generation prompt ---
template_decomposition = """
You are a helpful assistant that generates multiple sub-questions related to an input question. \\n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \\n
Generate multiple search queries related to: {question} \\n
note: generate only queries
Output (only 3 queries): 
"""

prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)
```

## 2\. **Query Generation Chain**

a chain do following things

1. `prompt_decomposition` - take input prompts.
    
2. `llm` - run it through llm.
    
3. `StrOutputParser()` - parse the llm response into string
    
4. `(lambda x : x.split("\n") )` - split the parse output into list of sub-questions
    

```python
# --- Query Generation Chain ---
query_genration = ( prompt_decomposition | llm | StrOutputParser() | (lambda x : x.split("\\n") ) ) 
```

## 3\. **Prompt for Sub-question Answering**

* `reterived_template` - Prompt instruction to answer sub-question using retrieved context.
    
* `reterived_prompt` - it is used in chain
    

```python
# --- Reterived sub-quetion ans generation prompt ---
reterived_template ="""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\\n
Question: {question} \\n
Context: {context} \\n
Answer:
"""
reterived_prompt = ChatPromptTemplate.from_template(reterived_template)
```

## 4\.  **Sub-question Retrieval and Answer Generation**

### 1\. **Q&A Pair Formatting Utility.** `format_qa_pairs`

Formats each `sub-question` and its answer into a readable string, ready to be used as context for the final answer.

### 2\. **Sub-question Retrieval and Answer Generation.** `retrieve_and_rag`

`retrieve_and_rag` this is main function that do followings things:

1. `sub_questions` **Decompose** the main question into sub-questions using `query_genration`.
    
2. For each `sub_question`:
    
    1. **Retrieve** relevant context from the vector store.
        
    2. **Run a chain** that feeds the sub-question and its context to the LLM to get an answer.
        
    3. **Collect** all answers.
        
3. **Format** all Q&A pairs into a single context string using `format_qa_pairs` utility.
    

```python
# --- Utility: contenxt merging subquetion with answers ---
def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\\nAnswer {i}: {answer}\\n\\n"
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
```

## 5\. **Final Synthesis Prompt and Chain**

* prompt template - prompt instruction for llm to answer the question by using all the Q&A pairs as context.
    
* `final_rag_chain` - run a prompt through llm and parse the output.
    
* **Execution -** Runs the final synthesis chain to get a concise, synthesized answer.
    

```python
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
```

# 2\. Iterative Query Decomposition

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">GitHub code file - <a target="_self" rel="noopener noreferrer" class="notion-link-token notion-focusable-token notion-enable-hover" href="https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Query-Decomposition/main.py" style="pointer-events: none">link</a></div>
</div>

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759836680777/04cd6be6-b782-4de1-ac0e-00636aa7d281.png align="center")

## 1\. **Prompt**

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">this is same step as we discussed in <a target="_self" rel="noopener noreferrer nofollow" href="https://www.notion.so/Query-Translation-Query-Decomposition-Code-Explanation-284af752ca2c8034bc9df1131a7de2b6?pvs=21" style="pointer-events: none">Parallel Query Decomposition</a></div>
</div>

* [`template_decomposition` - a prompt in](https://www.notion.so/Query-Translation-Query-Decomposition-Code-Explanation-284af752ca2c8034bc9df1131a7de2b6?pvs=21)struction that tells llm to break down [complex questions into 3 sub-questions](https://www.notion.so/Query-Translation-Query-Decomposition-Code-Explanation-284af752ca2c8034bc9df1131a7de2b6?pvs=21)
    
* `prompt_decomposition` - `template_decomposition` it wraps around `prompt_decomposition` to u[se it in chain](https://www.notion.so/Query-Translation-Query-Decomposition-Code-Explanation-284af752ca2c8034bc9df1131a7de2b6?pvs=21)
    

```python
# --- Query generation prompt ---
template_decomposition = """
You are a helpful assistant that generates multiple sub-questions related to an input question. \\n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \\n
Generate multiple search queries related to: {question} \\n
note: generate only queries
Output (only 3 queries): 
"""

prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)
```

## 2\. **Query Generation Chain**

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">this is also same step as we discussed in <a target="_self" rel="noopener noreferrer nofollow" href="https://www.notion.so/Query-Translation-Query-Decomposition-Code-Explanation-284af752ca2c8034bc9df1131a7de2b6?pvs=21" style="pointer-events: none">Parallel Query Decomposition</a></div>
</div>

Query Generation Chain do following things

1. `prompt_decomposition` - take input prompts.
    
2. `llm` - run it through llm.
    
3. [`StrOutput`](https://www.notion.so/Query-Translation-Query-Decomposition-Code-Explanation-284af752ca2c8034bc9df1131a7de2b6?pvs=21)`Parser()` - parse the llm response into string.
    
4. `(lambda x : x.split("\\n") )` - split the parse output into list of sub-questions.
    

```python
# --- Query Generation Chain ---
query_genration = ( prompt_decomposition | llm | StrOutputParser() | (lambda x : x.split("\\n") ) )
```

## 3\. **Prompt for Sub-question Answering**

A prompt template that asks the LLM to answer a sub-question using:

* The sub-question itself,
    
* Any background Q&A pairs,
    
* Additional retrieved context.
    

```python
# --- Reterived sub-quetion ans generation prompt ---
reterived_template = """Here is the question you need to answer:

\\n --- \\n {question} \\n --- \\n

Here is any available background question + answer pairs:

\\n --- \\n {q_a_pairs} \\n --- \\n

Here is additional context relevant to the question: 

\\n --- \\n {context} \\n --- \\n

Use the above context and any background question + answer pairs to answer the question: \\n {question}
"""
reterived_prompt = ChatPromptTemplate.from_template(reterived_template)
```

## 4\. **Sub-question Retrieval and Answer Generation**

### 1\. **Q&A Pair Formatting Utility.** `format_qa_pairs`

Formats a `sub_question` and its answer into a readable string.

### 2\. **Sub-question Retrieval and Answer Generation.** `retrieve_and_rag`

`retrieve_and_rag` this is main function that do followings things:

1. `sub_questions` **Decompose** the main question into sub-questions using `query_genration`.
    
2. For each `sub_question`:
    
    1. **Retrieve** relevant context from the vector store.
        
    2. **Run a chain** that feeds the sub-question, any background Q&A pairs, and the retrieved context to the LLM to get an answer.
        
    3. **Format** the Q&A pair and accumulate it.
        
3. Returns all Q&A pairs as a single formatted string.
    

```python
# --- Utility:contenxt merging subquetion with answers ---
def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\\nAnswer: {answer}\\n\\n"
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
        q_a_pairs = q_a_pairs + "\\n-----\\n" + q_a_pair
    
    return q_a_pairs

question = 'what is js'
context = retrieve_and_rag(question,query_genration,reterived_prompt)
```

## 5\. **Final Synthesis Prompt and Chain**

* prompt template - prompt instruction for llm to answer the question by using all the Q&A pairs as context.
    
* `final_rag_chain` - run a prompt through llm and parse the output.
    
* **Execution -** Runs the final synthesis chain to get a concise, synthesized answer.
    

```python
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
```

# Resources

## **1.Github code**

* [**Parallel** Query **Decomposition**](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Parallel-Decomposition/main.py)
    
* [Iterative Query Decomposition](https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Query-Decomposition/main.py)
    

## 2\. **Articles**

* [What is Query Decomposition](https://onkark.hashnode.dev/query-decomposition)
    

---

Learned something? Hit the ❤️ to say “thanks!” and help others discover this article.

**Check out** [**my blog**](https://onkark.hashnode.dev/series/genai-devlopment) **for more things related GenAI.**