# Malibu — Building an Autoregressive Transformer from Scratch

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Research Platform](https://img.shields.io/badge/Research_Site-malibu--web.vercel.app-0ea5e9.svg?style=flat-square)](https://malibu-web.vercel.app)
[![Status](https://img.shields.io/badge/Status-In%20Development-blue.svg?style=flat-square)](#)

**Malibu** is an open-source study project building a causal autoregressive Transformer language model from first principles in raw PyTorch—no high-level wrapper libraries (no Hugging Face `Trainer`, no LangChain), just raw tensor operations, attention math, and model architecture.

Created by **Bradley Joe (Joseph)** (Second-Year MEng Computer Science with AI @ University of York).

---

## 🎯 Project Goals

- Understand and implement the core mathematical components of modern Large Language Models (LLMs).
- Write custom tokenization, multi-head self-attention, and training loops from scratch.
- Train locally on Apple Silicon (MPS GPU) with zero API costs.
- Deploy the resulting model with a production REST API and integrate cloud artifact management via AWS (S3 & App Runner).

---

## 🗺️ Construction Roadmap

- [ ] **Phase 1: Tokenizer from Scratch**
  - Text-to-integer mappings (`stoi` & `itos`).
  - Vocabulary serialization and round-trip verification.
- [ ] **Phase 2: Embeddings & Coordinates**
  - Token embedding layers (`nn.Embedding`).
  - Positional embeddings (giving sequences order).
- [ ] **Phase 3: Multi-Head Causal Self-Attention**
  - Query, Key, and Value linear projections.
  - Scaled dot-product attention calculation.
  - Causal masking to prevent future token leakage.
  - Multi-head splitting and recombination.
- [ ] **Phase 4: Transformer Blocks & Training Engine**
  - RMSNorm / Pre-LayerNorm stabilization.
  - Position-wise Feed-Forward Networks (GELU activation).
  - Cross-entropy loss and AdamW optimization with cosine learning rate decay.
- [ ] **Phase 5: Inference & Text Generation**
  - Autoregressive generation loop.
  - Temperature, Top-K, and Top-P (Nucleus) sampling.
- [ ] **Phase 6: Cloud Serving & AWS MLOps**
  - FastAPI inference server with Swagger documentation.
  - Model checkpoint registry in Amazon S3.
  - Containerization and deployment to AWS.

---

## 🛠️ Stack & Prerequisites

- **Language**: Python 3.12
- **Deep Learning Framework**: PyTorch (raw tensors & `nn.Module`)
- **Hardware Acceleration**: Apple Silicon MPS (Metal Performance Shaders) / CUDA / CPU
- **Testing**: `pytest`
 
