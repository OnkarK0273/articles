---
title: "Query Translation (Query Re-writing) [Advance RAG]"
seoTitle: "what is Query Transformation"
seoDescription: "The Query transformation addresses the fundamental issue in native RAG system i.e. quality of retrieved documents from vectorDB is depends solely on user qu"
datePublished: Thu Sep 25 2025 14:07:23 GMT+0000 (Coordinated Universal Time)
cuid: cmfzhm1ow000002jvg8og0coc
slug: query-translation-query-re-writing-advance-rag
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1758890445483/2eb691ea-e93c-488a-887a-51ea69e76525.png
tags: rag, advanced-rag, query-transformation

---

We will now explore some **advanced techniques, patterns, and concepts** related to **RAG**. if you don’t or less knowledge about RAG, check it out blog first - [link](https://onkark.hashnode.dev/how-rag-makes-llms-smarter) unless you can’t get topic.

# The problem

The Query Translation addresses the fundamental issue in native RAG system i.e. quality of retrieved documents from vectorDB is depends solely on user query, if user query is complex or ambiguous that affect the quality of retrieved documents.

> Bad query = Degrading retrieval quality

so, to address that Query Translation comes into the picture

# What is Query Translation

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758806881908/c79133db-2e5f-4f81-9d63-291f0815b7a5.png align="center")

* Query Translation means re-writing the user query to enhance its relevancy so it’s helpful to retrieve those with the highest semantic similarity from vectorDB
    
* Query Translation techniques apply before retrieving the document
    

> re-write query = elevating retrieval quality

# Query Translation **Methods**

now we are going to see most common **Query Translation Techniques** are used in RAG

## 1\. Parallel (Fan-out) Retrieval

**Parallel (Fan-out) Retrieval**, also known as **Multi-Query Retrieval**, is a **Query Translation** technique in RAG that aims to increase the **recall** (breadth) and **diversity** of retrieved documents, especially for complex or ambiguous queries.

It is a straightforward yet highly effective way to overcome the limitation of relying on a single vector embedding for search.

### How Parallel (Fan-out) Works Step-by-Step

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758807585957/1738f6bc-1149-4670-b2be-e86893b17a85.png align="left")

1. ### User input
    

* First user enters query
    

2. ### Query Generation
    

* we don’t know about user query, he can ask something crazy
    
* if the user query is complex or ambiguous which may degrade the quality of retrieved documents from the vector database
    
* so, to tackle the query issue we have to generate more than one query by using llm-1.
    

3. ### Parallel Retrieval
    

* We embed each generated query and perform parallel retrieval from the vector database.
    
* for each generated query it retrieves relevant documents from vectorDB
    

4. ### Context Merging
    

* We merge the retrieved documents from all queries and filter out duplicates to retain only unique context.
    

5. ### Answer Generation
    

* after getting unique filter out document we send it to llm-2 with original user query
    
* because of more relevant context, LLM-2 can generate a more accurate and complete response to the user’s original query.
    

---

### The problem in Parallel (Fan-out) Retrieval

* in large scale production grade RAG, there is high chance retrieving 30+ document during context merging state.
    
* sending this large context to llm with user query it insufficient for context window imposed by llm
    
* so, to address this issue we use another query translation techniques i.e. **Reciprocal Rank Fusion**
    

## 2\. Reciprocal Rank Fusion (RRF)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758868223733/3f8b13cf-a348-4a8b-9a36-b3f7a35511cb.png align="center")

To address the issue in **Parallel (Fan-out) Retrieval** we use **Reciprocal Rank Fusion** Technique

### User input, Query Generation, Parallel Retrieval

> The above steps remain the same as discussed in the **Parallel (Fan-out) Retrieval** section.

### Context Merging

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758803890253/d9760cab-8d7e-484e-905b-e64de36d85d1.png align="left")

* The main difference lies in the **Context Merging** phase.
    
* Here, we apply the **RRF formula** to rank and select the **top 3 retrieved documents**.
    

### How RRF Works

In parallel Retrieval phase we got retrieved doc as per below

* **Line 1**: green doc. (1), orange doc (2), blue doc (3)
    
* **Line 2**: purple doc (1), blue doc (2), yellow doc (3)
    
* **Line 3**: blue doc (1), orange doc (2), green doc (3)
    

RRF Formula:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758804076750/a75830b0-ff9b-44db-95b1-5a7c38e15a65.png align="left")

Where:

k: a constant (typically 60)

ri(d): rank of document in the i-th ranked list

Now we apply formula to each unique document

1. **Blue Document Rank:**
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758804299134/bb2cd6f2-4e5a-48c0-9ede-1a1de891257a.png align="center")
    
2. **Green Document Rank:**
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758804312593/d3de5dbf-73dc-4808-999a-bf7710fa81b7.png align="center")
    
3. **Orange Document Rank:**
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758804326031/29dcbbdb-126b-4a23-a0cd-b9ce54ecc58f.png align="center")
    
4. **Purple Document Rank:**
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758804336544/5056f423-a8dd-4b20-9461-2aca824f2426.png align="left")
    
5. **Yellow Document Rank:**
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758804347651/6b2e2370-dec8-4c4d-96be-f1fabdc030d4.png align="left")
    

**Final RRF Scores (sorted):**

| **Item** | **RRF Score** |
| --- | --- |
| blue | 0.04839 |
| green | 0.03226 |
| orange | 0.03226 |
| purple | 0.01639 |
| yellow | 0.01587 |

**Final RRF ranking is:**

**blue &gt; green = orange &gt; purple &gt; yellow**

## 3\. Hypothetical Document Embeddings (**HyDE**)

The core idea of HyDE is to bridge the semantic gap between the short, often vague user query and the long, descriptive documents in the knowledge base. It does this by creating a "middleman" document.

In a traditional RAG system, the vector database stores embeddings of **answers** (document chunks), but the query is a **question**. HyDE shifts the search alignment from:

```plaintext
                           Question Embedding <-----> Answer Embedding
                                              to
                     Hypothetical Answer Embedding <-----> Real Answer Embedding
```

Since the hypothetical answer is much richer and more descriptive than the short query, its vector is much more likely to accurately land in the correct semantic "neighborhood" of the real documents.

### How HyDE Works Step-by-Step

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758890215375/6e7e130a-d6d3-4a1f-9e5e-1182ceed35b9.png align="center")

### Step 1: User Query Input

* **Action**: The user asks a question to the system.
    
* **Diagram**: The process starts with the **User Query** box containing the question: "What are the common side effects of that new vaccine?"
    

### **Step 2: Hypothetical Document Generation (The HyDE Step)**

The original query is *translated* into a detailed, hypothetical answer.

* **Input to LLM**: The user's short, semantic query is passed to an initial **LLM**.
    
* **LLM Action**: The LLM uses its vast internal knowledge (not the vector store yet) to generate a detailed, plausible **Hypothetical Document** that *would* answer the query, often using technical or domain-specific language.
    
* **Diagram**: The LLM outputs the "Generated ans. (Hypothetical document)": *"The new vaccine developed by Acme Pharma is known to cause mild, temporary side effects, including localized pain at the injection site, fatigue lasting up to 24 hours, and a slight fever."*
    

### **Step 3: Embedding Creation**

The system prepares the hypothetical document for search.

* **Action**: The long, descriptive **Hypothetical Document** is converted into a **vector embedding** (a numerical representation).
    
* **Diagram**: The hypothetical document flows into the **Embedding** block.
    

### **Step 4: Vector Search (Answer-to-Answer Matching)**

The embedding of the hypothetical answer is used to find the real, factual answers.

* **Action**: The hypothetical embedding is used to search the **Vector Database**. Because the embedding is long and rich (like a real document), it performs a much more precise **semantic search** than the short original query could.
    
* **Result**: The system retrieves the actual, authoritative document chunks that are closest to the vector of the hypothetical answer.
    
* **Diagram**: The **Embedding** flows into the **Vector Database**, resulting in **Retrieved Docs**.
    

### **Step 5: Final Response Generation (Synthesis)**

The system generates the final, grounded answer.

* **Inputs to Final LLM**: This final LLM receives two critical inputs:
    
    1. The **Original User Query** (to know what question to answer).
        
    2. The **Retrieved Docs** (to ensure the answer is factual and grounded).
        
* **LLM Action**: The LLM synthesizes the information from the **Retrieved Docs** to formulate a cohesive, factual, and helpful **Response** to the user's question. It disregards any potential inaccuracies in the initial hypothetical document, as the retrieved real documents are the single source of truth.
    
* **Diagram**: The **Retrieved Docs** and the original **User Query** (implicitly or via the connecting line in the corrected flow) are fed into the final LLM to produce the **Response**.
    

### Answer Generation

* We provide the **top 3 ranked documents** to **LLM-2**, along with the **original user query**.
    
* Since we reduce the number of documents from 5 to 3, this helps **avoid exceeding the context window** limits of the LLM.
    
* As a result, **LLM-2 receives more relevant context**, which leads to a **more accurate and focused response**.
    

# Resources

1. langChain academy - [GitHub](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb)
    
2. langChain academy - [Youtube-01](https://www.youtube.com/watch?v=JChPi0CRnDY&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=5), [Youtube-02](https://www.youtube.com/watch?v=SaDzIVkYqyY)