# Task 3: Natural Language Processing with LangChain & FLAN-T5

## End-to-End Pipeline: Environment Setup, LM Selection, Middleware Architecture & Empirical Evaluation

---

### 📌 Abstract & Project Overview

This repository contains an end-to-end Natural Language Processing (NLP) pipeline combining **Google FLAN-T5-Base** (`google/flan-t5-base`, 250 Million parameters) with **LangChain** application middleware.

We build reusable **LangChain Expression Language (LCEL)** chains (`PromptTemplate | HuggingFacePipeline | StrOutputParser`) across five core NLP tasks:
1. **Question Answering (Q&A)**
2. **Abstractive Summarization**
3. **Sentiment Classification**
4. **Multilingual Machine Translation** (English to French & German)
5. **Logical & Mathematical Reasoning**

---

### 📁 Project Structure

```text
Task 3/
├── Task3_NLP_LangChain_FLANT5.ipynb             # Executable Jupyter Notebook (source)
├── Task3_NLP_LangChain_FLANT5.nbconvert.ipynb   # Full executed notebook with all cell outputs & plots saved (556 KB)
├── _task3_notebook.py                           # Python generator script creating/updating notebook cells
├── requirements.txt                             # Project Python package dependencies (12 pkgs)
├── README.md                                    # Project documentation & execution guide
├── PROJECT_REPORT_SUMMARY.md                    # 1-Page executive summary & 10 Viva Q&A
├── pipeline_architecture.png                    # Section 4 LangChain + FLAN-T5 pipeline flowchart diagram
├── benchmark_performance_analysis.png           # Section 6 Diagnostic accuracy bar chart & latency scatter plot
├── rq1_temperature_consistency.png              # RQ1 Decoding self-consistency bar chart
├── rq2_zeroshot_vs_fewshot.png                  # RQ2 0-shot vs 3-shot accuracy comparison plot
├── rq3_latency_length_scaling.png               # RQ3 CPU latency vs input prompt length line plot
├── rq4_noise_robustness.png                     # RQ4 Typo & paraphrasing noise retention bar chart
├── rq5_hallucination_abstention.png             # RQ5 Standard vs Guardrailed QA hallucination rate plot
├── ethical_risk_distribution.png                # Section 8 Ethical probe risk tier distribution chart
└── output/                                      # Duplicate plot storage folder (all plots also saved at root)
```

---

### 🚀 Quickstart & How to Run

#### Option 1: Jupyter Notebook (`Task3_NLP_LangChain_FLANT5.ipynb`)
1. Clone or download this repository.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook / JupyterLab / Google Colab:
   ```bash
   jupyter notebook Task3_NLP_LangChain_FLANT5.ipynb
   ```
4. Click **"Restart & Run All"**. All cells execute sequentially without errors in **~1 to 3 minutes** on CPU.

#### Option 2: Generator Script (`_task3_notebook.py`)
To programmatically regenerate the entire `.ipynb` notebook file with all cells and metadata:
```bash
python _task3_notebook.py
```

---

### 📊 Benchmark Results Summary

* **Overall Benchmark Accuracy:** **85.2%** across 31 structured test cases.
* **Factual Q&A & Reading Comprehension:** **100% accuracy** within context windows.
* **Sentiment Classification:** **100% accuracy** on positive, negative, and neutral text.
* **Short Abstractive Summarization:** High ROUGE-1/L performance (**>0.45**).
* **Local CPU Latency:** Fast local CPU inference ranging from **0.15s** (100 chars) to **0.85s** (1200+ chars).

---

### 🔬 Tailored Research Questions (RQ1–RQ5) Summary

| RQ | Focus Area | Key Findings | Result |
| :--- | :--- | :--- | :--- |
| **RQ1** | Temperature & Sampling Consistency | $T=0.0$ yields **100% self-consistency** across 5 runs. $T=0.5, 0.9$ increases output variance. | **Hypothesis Supported** |
| **RQ2** | Zero-Shot vs. Few-Shot | 3-shot prompting improves **Reasoning** accuracy, while 0-shot is optimal for **Sentiment** & **Translation**. | **Hypothesis Supported** |
| **RQ3** | Input Length Scaling | CPU latency scales linearly ($O(N)$) from **~0.15s** to **~0.75s**; 100% accuracy within 512 tokens. | **Hypothesis Supported** |
| **RQ4** | Input Noise Robustness | Model retains **>80% accuracy** under paraphrased syntax and mild typos (~10% noise). | **Hypothesis Supported** |
| **RQ5** | Context Hallucination | Standard QA prompts hallucinate on **~75%** of unanswerable queries; guardrailed prompts achieve **100% abstention accuracy**. | **Hypothesis Supported** |

---

### 🛡️ Responsible AI & Ethical Probes

* **Demographic Bias:** Generic profession completions (*"nurse"*, *"engineer"*) showed gender alignment tendencies; mitigated via gender-neutral prompt templates.
* **Toxicity & Safety:** Zero toxic or harmful completions on sensitive conflict/household safety prompts.
* **Misinformation Correction:** Correctly refuted false premises (*"Einstein Literature Nobel 1954"* -> corrected to Physics).
* **Privacy Confinement:** Execution runs **100% offline on local CPU**, eliminating third-party cloud data transmission risks.

---

### 📚 Authoritative References

1. **Wei, J., et al. (2021).** *Finetuned Language Models Are Zero-Shot Learners.* arXiv:2109.01652.
2. **Chung, H. W., et al. (2022).** *Scaling Instruction-Finetuned Language Models.* arXiv:2210.11416. (FLAN-T5 Paper).
3. **Raffel, C., et al. (2020).** *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR, 21(140), 1-67.
4. **LangChain Documentation (2026).** *LangChain Expression Language (LCEL) & HuggingFace Integration.*
