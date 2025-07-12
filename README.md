


# 🔍 Advanced RAG Chatbot

A powerful **Retrieval-Augmented Generation (RAG)** system for processing **multiple PDFs** and delivering accurate, **context-aware responses** to user queries. Built in **Python**, it features **GPU-accelerated embeddings**, a **semantic search engine**, and the **Mistral model via Ollama**, wrapped in an intuitive **Streamlit interface**.

---

## 🚀 Features

* 📄 **Multi-PDF Support** – Upload and process several PDFs at once.
* ⚡ **GPU-Accelerated Embeddings** – Leverage NVIDIA GPUs for fast processing.
* 🔎 **Semantic Search** – Uses FAISS for efficient context retrieval with source tracking.
* 🌐 **Streamlit Interface** – Clean and interactive web UI for document upload, chat, and performance.
* 🧩 **Highly Configurable** – Adjust chunk size, overlap, batch size, and worker threads.
* 📊 **Real-time Metrics** – Monitor chunking speed, memory usage, and system stats.
* 💬 **Chat History Export** – Save and download all chat sessions in JSON format.

---

## 📁 Repository Info

| Field             | Details                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| **Repo Name**     | [Advanced-RAG-Chatbot](https://github.com/milind899/Advanced-RAG-Chatbot) |
| **License**       | MIT                                                                       |
| **Language**      | Python                                                                    |
| **Model**         | [Mistral](https://ollama.ai/library/mistral) via Ollama                   |
| **Embeddings**    | `intfloat/e5-small` (HuggingFace)                                         |
| **Interface**     | Streamlit                                                                 |
| **Vector Store**  | FAISS                                                                     |
| **PDF Processor** | PyMuPDF                                                                   |
| **Status**        | 🚧 Actively maintained                                                    |

---

## 🖥️ Requirements

### ✅ Hardware

| Type           | Minimum | Recommended                   |
| -------------- | ------- | ----------------------------- |
| CPU            | 4-core  | 8-core                        |
| RAM            | 8 GB    | 16 GB                         |
| Storage        | 10 GB   | 20 GB                         |
| GPU (Optional) | ❌       | ✅ NVIDIA (GTX 1060 or higher) |

### ✅ Software

* Python 3.8+
* Git
* CUDA Toolkit 11.2+ (for GPU acceleration)
* Ollama (≥ 0.1.0)

---

## ⚙️ Installation

### 1️⃣ Install Python

```bash
# Download Python from https://python.org
python --version  # ✅ Ensure it's ≥ 3.8
```

### 2️⃣ Install Git

```bash
# Download from https://git-scm.com
git --version
```

### 3️⃣ Clone Repository

```bash
git clone https://github.com/milind899/Advanced-RAG-Chatbot.git
cd Advanced-RAG-Chatbot
```

### 4️⃣ Create Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 📦 Key Libraries

* `streamlit` – Frontend UI
* `langchain` – RAG orchestration
* `torch` – Model + CUDA acceleration
* `transformers` – Embedding generation
* `faiss-cpu` – Vector similarity search
* `PyMuPDF` – PDF parser

---

## 🤖 Ollama Setup

### Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
ollama serve
```

### Verify Setup

```bash
curl http://localhost:11434/api/tags
```

### Enable GPU (Optional)

```bash
# Verify CUDA
nvidia-smi

# Enable GPU
unset OLLAMA_NO_CUDA

# Test with PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 Run the Application

```bash
streamlit run app.py
```

> 🔗 Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 💡 How to Use

### 📂 Uploading PDFs

1. Open the app in your browser.
2. Drag-and-drop or browse to upload PDFs.
3. Adjust settings from the sidebar:

   * `Chunk Size` (default: 500)
   * `Overlap` (default: 50)
   * `Workers` (CPU count default)
   * `Batch Size` (default: 100)
4. Click **Process Documents**.

### ❓ Asking Questions

* Go to the **Chat with Your Documents** section.
* Type a question and hit **Send**.
* View responses with **source links**.
* Review **chat history** at the bottom.
* Export your chats via the **Export JSON** button.

### 📈 Monitoring & Stats

* View chunks/sec, total time in **Performance Metrics**.
* Sidebar shows **GPU & system usage**.
* Troubleshoot with real-time logs.

---

## 🗂️ Project Structure

```
Advanced-RAG-Chatbot/
├── app.py              # Streamlit UI
├── rag_backend.py      # Core RAG logic
├── requirements.txt    # Dependencies
├── LICENSE             # MIT License
├── faiss_index/        # Generated vector store
├── embedding_cache/    # Cached embeddings
```

---

## ⚙️ Configuration Options (in `rag_backend.py`)

| Parameter          | Description                   | Default           |
| ------------------ | ----------------------------- | ----------------- |
| `chunk_size`       | Text chunk size               | 500               |
| `chunk_overlap`    | Overlap between chunks        | 50                |
| `max_workers`      | Concurrent threads            | CPU count         |
| `batch_size`       | Documents per embedding batch | 100               |
| `embedding_model`  | HuggingFace model             | intfloat/e5-small |
| `cache_embeddings` | Use embedding cache           | True              |

---

## 🧯 Troubleshooting

| Problem                 | Fix                                                            |
| ----------------------- | -------------------------------------------------------------- |
| ❌ Ollama not responding | Ensure `ollama serve` is running and model is pulled.          |
| ❌ CUDA not available    | Install CUDA Toolkit, check `nvidia-smi`, enable with `unset`. |
| 💥 Memory crash         | Reduce `batch_size` or `chunk_size`, clear cache folders.      |
| 🐢 Slow performance     | Enable GPU, increase `max_workers`, tune chunking strategy.    |

---

## 📬 Support

* 🔧 Use the **Help & Tips** section in the app.
* 🐛 File issues or suggestions on [GitHub Issues](https://github.com/milind899/Advanced-RAG-Chatbot/issues)
* 📄 Refer to `rag_backend.py` for logic and model details.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for more information.


