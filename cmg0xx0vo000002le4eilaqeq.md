---
title: "Query Translation (Query Decomposition) [Advance RAG]"
seoTitle: "what is Query Decomposition"
seoDescription: "Query decomposition is advance RAG techniques used to breaking down complex problem into multiples sub-problems, basically we make less abstract problem."
datePublished: Fri Sep 26 2025 14:31:35 GMT+0000 (Coordinated Universal Time)
cuid: cmg0xx0vo000002le4eilaqeq
slug: query-decomposition
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1758893146131/6c65c0a9-b381-49da-b663-381560f52082.png
tags: genai, advanced-rag, query-transformation, query-decomposition, query-translation

---

if you don’t or less knowledge about RAG, check it out blog first - [link](https://onkark.hashnode.dev/how-rag-makes-llms-smarter) unless you can’t get topic

# What is Query Decomposition

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758893255852/c9caec70-4c2f-47f8-81d1-cfe21505a86c.png align="center")

Query decomposition is advance RAG techniques used to breaking down complex problem into multiples sub-problems, basically we make less abstract problem. because of multiple sub-problem we get more relevant chunk from retrieval that improve accuracy of llm response.

# Decomposition Methods

now we are going to see most common **Query Decomposition Techniques** are used in RAG.

## 1\. Parallel Decomposition

This methods is used when components of complex query is **independent** to each other. goal is to maximize retrieval context and efficiency.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758895613743/3b9d6c0b-4aa6-497b-a529-df228eaff440.png align="center")

### Step 1: User Query

* **What it is**: This is the starting point. A user asks a complex question that may contain multiple sub-topics or require information from various sources to be answered completely. In your diagram, the user query is "What is Machine learning".
    

### Step 2: Query Decomposition

* **What it is**: The user's query is sent to a large language model (LLM). The LLM's job is to analyze the complex question and break it down into smaller, simpler, and **independent** sub-questions.
    
* **What the diagram shows**: The LLM takes "What is Machine learning" and generates three separate sub-queries: "What is learning", "What is Machine", and "What is Machine learning". This is an example of the LLM trying to get a more comprehensive set of facts by querying different aspects of the original topic.
    

### Step 3: Parallel Retrieval

* **What it is**: Each of the generated sub-queries is sent simultaneously to a vector database. This is a critical step that makes this method efficient. The system does not wait for one sub-query to be answered before starting the next.
    
* **What the diagram shows**: Three separate arrows leave the LLM, each with a different query, and they all go to the "Vector Database." The vector database's role is to perform a similarity search for each query and retrieve the most relevant documents (or "chunks") from its indexed knowledge base. For example:
    
    * `Generated query 1`: "What is learning" retrieves documents about the general concept of learning.
        
    * `Generated query 2`: "What is Machine" retrieves documents about the concept of a machine.
        
    * `Generated query 3`: "What is Machine learning" retrieves core definitions and descriptions of the field itself.
        

### Step 4: Context Aggregation

* **What it is**: The documents retrieved for each sub-query are collected. Since this is a parallel process, the system now has a broader and more comprehensive set of information than it would have from a single search.
    
* **What the diagram shows**: The retrieved documents from all three queries (`Query res 1`, `Query res 2`, `Query res 3`) are combined and sent to the LLM. This gives the LLM a rich, multi-faceted context to work with.
    

### Step 5: Final Response Generation

* **What it is**: The LLM takes all the retrieved context (from all the sub-queries) and the original user query as its input. Its final task is to read, understand, and synthesize all this information into a single, coherent, and detailed answer.
    
* **What the diagram shows**: The LLM processes the combined context and generates a "Final Response" that directly and comprehensively addresses the user's original question, "What is Machine learning."
    

This complete process ensures that the system doesn't miss any critical information and can provide a more accurate and well-rounded answer than a basic RAG system that would only perform a single search on the original query.

## 2\. Iterative/Multi-hop Query Decomposition

This method is used when components of a complex query require **sequential reasoning**, where one step is completed **based on the previous step's completion**.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758895756964/58adb5ad-2619-4d3c-a1b0-51daa3232415.png align="center")

### Step 1: Initial Decomposition (Hop 1)

The goal is to find the first dependent piece of information.

* **Input to LLM**: The original complex **User Query**.
    
* **LLM Action**: The LLM analyzes the question and generates the first logical sub-question.
    
* **Generated Query 1**: "Who invented the light bulb?"
    
* **Retrieval**: The system searches the **Vector Database** using Query 1.
    
* **Output (Retrieved Docs 1)**: The system finds and extracts the fact: **(Thomas Edison)**.
    

### Step 2: Contextual Refinement (Hop 2)

The system uses the new fact to narrow the search. This is the **dependency** that defines multi-hop.

* **Input to LLM**: The **User Query** is combined with the **first retrieved fact (Thomas Edison)**.
    
* **LLM Action**: The LLM uses this combined context to formulate the next logical, *specific* query.
    
* **Generated Query 2**: "What country was **Thomas Edison** born in?" (Notice how the name from Hop 1 is now in the query).
    
* **Retrieval**: The system searches the **Vector Database** using Query 2.
    
* **Output (Retrieved Docs 2)**: The system finds and extracts the fact: **(United States)**.
    

### Step 3: Final Targeted Retrieval (Hop 3)

The system uses the results of the second hop to generate the final search query.

* **Input to LLM**: The User Query is now combined with **two retrieved facts (Thomas Edison + United States)**.
    
* **LLM Action**: The LLM formulates the final search query to get the ultimate answer.
    
* **Generated Query 3**: "What is the capital of the **United States**?"
    
* **Retrieval**: The system searches the **Vector Database** using Query 3.
    
* **Output (Retrieved Docs 3)**: The system finds and extracts the final fact: **(Washington, D.C.)**.
    

### Step 4: Final Synthesis

The LLM integrates all the pieces of the puzzle to generate a comprehensive final answer.

* **Input to LLM**: The final input box contains the original User Query **plus all three facts** (Thomas Edison + United States + Washington, D.C.).
    
* **LLM Action**: The LLM reads all the context and synthesizes it into a single, cohesive, and logically structured response.
    
* **Final Response**: "The inventor of the light bulb was Thomas Edison, who was born in the United States. The capital of that country is **Washington, D.C.**" 💡
    

# Resources

1. langChain academy - [GitHub](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb)
    
2. langChain academy - [Youtube](https://www.youtube.com/watch?v=h0OPWlEOank)
    

---

Learned something? Hit the ❤️ to say “thanks!” and help others discover this article.

**Check out** [**my blog**](https://onkark.hashnode.dev/series/genai-devlopment) **for more things related GenAI**