# 🚀 VectoEngine

A custom-built, zero-dependency Vector Database and Retrieval-Augmented Generation (RAG) engine built entirely from scratch in Node.js. 

This project intentionally bypasses high-level wrappers like LangChain, Pinecone, or ChromaDB to explore the raw mathematics of vector similarity, spatial partitioning (KD-Trees), graph routing (HNSW), and local LLM orchestration.

📄 **Read the Full Research Paper:** 
> *For a deeply detailed, handwritten breakdown of the mathematical formulas, time complexities, and algorithmic proofs behind this engine here:*  
> 👉 **https://drive.google.com/file/d/1cP0RBPnH38trG7rCQqcRDyRvEwZYPjtq/view?usp=drivesdk**

---

## ⚙️ Prerequisites & Installation

To run this engine completely locally (100% offline and private), you will need **Node.js** and **Ollama**.

### 1. Install Node.js
This project relies on the native `fetch` API for internal routing. You must have **Node.js v18.0 or higher**.
* Download and install from: [nodejs.org](https://nodejs.org/)

### 2. Install Ollama (Local AI)
Ollama runs the neural networks that power the text embeddings and the RAG chat generation.
* Download and install from: [ollama.com](https://ollama.com/)
* Once installed, open your terminal/command prompt and download the required models:
  # Download the Embedding Model (translates human text into mathematics)
```bash
ollama pull nomic-embed-text
```
  # Download the Generative Model (reasons and answers your questions)
```bash
  ollama pull llama3.2
```
### 3. Start the VectoScale Engine
Clone this repository to your machine, 
```bash
git clone https://github.com/subhadeep322/VectoEngine.git
```
then run the following commands in your terminal:
```bash
npm install
npm start
```
Open your web browser and navigate to http://localhost:8080 to view the interactive dashboard!
### 🎮 How to Use the Engine
The UI is divided into three interactive modules to demonstrate vector mathematics in real-time:
1. Visual Vector Search: Type a keyword (e.g., "Pizza" or "Math") to see how the engine translates text into a 16-Dimensional vector and searches the database. Toggle between algorithms (Brute Force, KD-Tree, HNSW) to see real-time microsecond speed differences in the Performance Metrics panel.
2. Manual DB Entry: Type a custom concept (e.g., Title: "Teen Titans", Category: "TV Show"). The engine will embed it and plot a new neon point on the spatial graph, updating the search space live.
3. The RAG Assistant: Upload a PDF or TXT file. The engine will chunk it, embed it, and store it. Switch to the "AI Assistant" tab to ask complex questions about your document, complete with verifiable mathematical citations.
### 🧠 High-Level Architecture
1. Data Ingestion: Documents are mathematically processed using a Sliding Window Chunking Algorithm (250-word blocks with a 30-word overlap) to mathematically guarantee that edge-boundary context is never destroyed.
2. Vectorization: nomic-embed-text translates these strings into high-dimensional latent space.
3. Search Algorithms:
#### Brute Force: 
O(N)linear baseline to guarantee exact
nearest-neighbor accuracy.
#### KD-Tree:
 O(logN) spatial partitioning utilizing bounding-sphere intersection pruning.
#### HNSW (Hierarchical Navigable Small World):
O(logN)
 multi-layered graph utilizing exponential decay probability and greedy beam-search routing.
Generation: Retrieved factual chunks are injected into a Zero-Shot Prompt and processed by llama3.2 to synthesize highly accurate, hallucination-free answers.
### 📊 Understanding Distance Benchmarks
When you search the database, the results include a mathematical "Distance" score. Here is how the engine interprets those numbers based on the selected metric:
1. Cosine Distance (Range: 0.0 to 2.0)
Measures the angle between two concepts. This is the gold standard for NLP and text AI.
0.000 = Exact match (0° angle). The semantic intent is identical.
0.100 - 0.400 = Highly relevant (e.g., "Pizza" finding "Burger").
1.000 = Unrelated / Orthogonal (90° angle).
2.000 = Exact opposite concepts (180° angle).
2. Euclidean Distance (Range: 0.0 to Infinity)
Measures physical straight-line distance (Pythagorean theorem). Because VectoScale normalizes the demo vectors, they sit on a 16D sphere.
0.000 = Exact spatial match.
1.000 - 1.500 = Moderately unrelated. (At ~1.414, normalized vectors are mathematically 90° apart).
2.000 = Maximum distance in normalized space (opposite sides of the sphere).
3. Manhattan Distance (Range: 0.0 to Infinity)
Measures grid-based block distance (L1 Norm). It sums the absolute differences across all 16 dimensions without taking the square root.
0.000 = Exact spatial match.
3.000 - 6.000+ = Standard variance for unrelated items. Manhattan numbers will always appear much larger than Euclidean because it calculates the longest physical path across the axes.
![alt text](<Screenshot 2026-05-08 215350.png>)
