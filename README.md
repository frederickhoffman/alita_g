<p align="center">
  <img src="assets/first_banner.png" width="600" alt="Alita-G Banner">
</p>

# Alita-G: Self-Evolving Generative Agent for Agent Generation 🚀

[![WandB](https://img.shields.io/badge/Weights_%26_Biases-Monitoring-orange)](https://wandb.ai/alita-g)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Alita-G is a state-of-the-art self-evolution framework that transforms general-purpose agents into domain experts. By systematically generating, abstracting, and curating **Model Context Protocol (MCP)** tools, Alita-G achieves SOTA performance on complex reasoning benchmarks while reducing computational costs.

> **"Alita-G attains 83.03% pass@1 on GAIA validation, establishing a new state-of-the-art result."** — *Alita-G Paper*

---

## 📊 Performance Benchmarks (GAIA Paper)

Alita-G establishes new SOTA results across complex reasoning benchmarks. The table below highlights the performance of the **Alita-G (3x)** configuration reported in the [research paper](https://arxiv.org/abs/2510.23601).

| Benchmark | Baselines (Avg) | **Alita-G Paper** | Implementation Status |
| :--- | :---: | :---: | :---: |
| **GAIA** (Val) | 55.15% | 83.03% | ⚙️ Framework Ready (Needs Evolution) |
| **PathVQA** | 52.00% | 60.00% | ⚙️ Framework Ready (Needs Evolution) |
| **HLE** | 24.00% | 33.00% | ⚙️ Framework Ready (Needs Evolution) |
| **Pass@3** (GAIA) | - | 89.09% | ⚙️ Framework Ready (Needs Evolution) |

> [!IMPORTANT]
> **Reproduction Note**: The results above are from the Alita-G paper using a fully evolved MCP Box. This repository provides the **complete framework** (Generation, Abstraction, Inference). To reproduce the 83% score, you must first run the **Evolution Phase** to populate the `mcp_box.json` with domain-specific tools, as the agent currently initializes with an empty toolset.

---

## 🛠️ Key Features

- **Self-Evolving MCP Box**: Automatically curates a repository of specialized tools.
- **RAG-Enhanced Selection**: Dynamic retrieval of task-relevant primitives.
- **MCP Abstraction**: Generalizes task-specific code into robust, reusable tools.
- **Efficiency**: Reduces mean tokens per example by ~15% relative to strong baseline agents.

---

## 🚀 Quick Start

### 1. Installation
Ensure you have [uv](https://github.com/astral-sh/uv) installed, then:
```bash
uv sync
```

### 2. Configure API Keys
Add your keys to a `.env` file:
```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
WANDB_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 3. Run Evaluation
Reproduce paper results on GAIA, PathVQA, or HLE:
```bash
# GAIA Validation
uv run python -m alita_g.eval --dataset GAIA --split validation

# PathVQA
uv run python -m alita_g.eval --dataset PathVQA --samples 100

# Humanity's Last Exam (HLE)
uv run python -m alita_g.eval --dataset HLE --samples 100
```

### 4. Interactive UI
Visualize the agent graph and execution flow using LangGraph:
```bash
langgraph dev
```

---

## 📖 Methodology

Alita-G operates in three distinct phases:
1. **Task-Driven Generation**: A master agent executes tasks and distills raw MCPs from successful trajectories.
2. **Abstraction & Consolidation**: Raw MCPs are refined via parameter generalization and documentation enhancement to form an **MCP Box**.
3. **Inference**: A specialized agent performs similarity-based retrieval to select the best tools for a given query.

---

## 📝 Citation
```bibtex
@article{qiu2025alitag,
  title={Alita-G: Self-Evolving Generative Agent for Agent Generation},
  author={Qiu, Jiahao and Qi, Xuan and Wang, Hongru and others},
  journal={arXiv preprint arXiv:2510.23601},
  year={2025}
}
```
