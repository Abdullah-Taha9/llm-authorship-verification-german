<div align="center">

# 🔍 Are LLMs capable of Authorship Verification on German Texts?

<p align="center">
  <strong>A comprehensive evaluation framework for Large Language Models on German authorship verification tasks</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-experiments">Experiments</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

[![Paper](https://img.shields.io/badge/📄_Paper-ACL_2024-red.svg)](https://your-paper-url-here)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your-notebook-url-here)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <strong>Authors:</strong><br>
  <a href="mailto:abdullah.al-labani@utn.de">Abdullah Al-Labani</a> •
  <a href="mailto:nitish.devranii@utn.de">Nitish Devrani</a> •
  <a href="mailto:timothy.leonard@utn.de">Timothy Leonard</a><br>
  <em>Department of Computer Science & Artificial Intelligence<br>
  University of Technology Nurnberg (UTN)</em>
</p>

</div>

---

## 📖 Overview

This repository contains the implementation and evaluation framework for our ACL research paper investigating the capability of Large Language Models to perform authorship verification on German texts. We provide a comprehensive comparison between English and German performance across multiple state-of-the-art LLMs.

**Authorship Verification (AV)** is the task of determining whether two given texts were written by the same author. While this has been extensively studied for English, limited work exists for German texts, particularly using modern LLMs.

<div align="center">
  <img src="authorship_verification/assets/Authorship_Verification_Processing.png" alt="Authorship Verification Pipeline" style="width: 800px; margin: 20px 0;"/>
  <p><em>Figure 1: Authorship Verification Processing Pipeline</em></p>
</div>

### 🎯 Research Questions

- **RQ1**: How well do LLMs perform on German authorship verification compared to English?
- **RQ2**: Which prompt strategies are most effective for authorship verification tasks?
- **RQ3**: How do different LLM architectures compare on this task?
- **RQ4**: What is the human baseline performance on this task?

### 🔬 Key Contributions

- **Comprehensive LLM Evaluation**: Systematic evaluation of 8 state-of-the-art LLMs (GPT-4, DeepSeek, etc.)
- **Cross-lingual Analysis**: Direct comparison of German vs English authorship verification performance
- **Prompt Engineering**: Investigation of different prompting strategies (biased vs linguistic features)
- **Human Evaluation**: GUI-based human annotation system with comparative analysis
- **Reproducible Framework**: Clean, configurable codebase for easy reproduction and extension

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-authorship-verification-german/AV

# Install dependencies
pip install -r requirements.txt
```

### API Key Setup

Choose one of the following methods to set up your API key:

**Method 1: Using api_key.txt file**
```bash
echo "your_openai_api_key" > api_key.txt
```

**Method 2: Using .env file**
```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

### Run Your First Experiment

```bash
# Run a basic German authorship verification experiment
python authorship_verification/authorship_verification.py --config configs/experiment.yaml
```

This will:
- Load 2 German text pairs from Amazon reviews
- Use GPT-4.1-mini with linguistic features prompting
- Save results to `results/german_lip_longest_gpt-4.1-mini/`

### Expected Output

```
✅ API key loaded from api_key.txt
Starting experiment with configuration:
  Dataset: amazon_review (german)
  Prompt: lip (german)
  Model: openai/gpt-4.1-mini
  Max samples: 2
Loaded top-2 longest pairs for german (by min_len).

Evaluating gpt-4.1-mini on 2 samples...
gpt-4.1-mini-experiment: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.07s/it]

Experiment completed!
Accuracy: 0.500
F1-Score: 0.667
Total cost: $0.0010
Results saved to: results/german_lip_longest_gpt-4.1-mini

============================================================
EXPERIMENT SUMMARY
============================================================
Model: gpt-4.1-mini
Language: german
Prompt: lip
Total Samples: 2
Correct: 1
Accuracy: 0.500
F1-Score: 0.667
Cost: $0.0010
```

---

## 🧪 Experiments

### Supported Models

| Model | Provider | Parameters | Context Length |
|-------|----------|------------|----------------|
| GPT-4o | OpenAI | ~1.8T | 128K |
| GPT-4.1 | OpenAI | ~1.8T | 128K |
| GPT-4.1-mini | OpenAI | ~8B | 128K |
| GPT-4.1-nano | OpenAI | ~1B | 128K |
| DeepSeek-R1 | DeepSeek | 671B | 128K |
| DeepSeek-V3 | DeepSeek | 671B | 128K |
| O3-mini | OpenAI | ~8B | 128K |

### Prompt Strategies

#### 1. Biased Prompt
Simple, direct prompting that may introduce bias:
```
Do these two texts have the same author? Answer with 'yes' or 'no'.
```

#### 2. LIP (Linguistic Features) Prompt
Encourages analysis of linguistic features:
```
Analyze the writing style, vocabulary, sentence structure, and other linguistic 
features to determine if these texts were written by the same author.
```

### Datasets

- **Amazon Reviews**: German/English product reviews from the authorship verification dataset
- **Wikipedia**: Planned support for Wikipedia articles (future work)

### Data Loading Strategies

- **Random**: Standard random sampling (default)
- **Longest**: Select text pairs with longest minimum length
- **Shortest**: Select text pairs with shortest maximum length  
- **Longest Balanced**: Longest pairs with balanced same/different author labels
- **Shortest Balanced**: Shortest pairs with balanced same/different author labels

---

## ⚙️ Configuration

### Basic Configuration

Create or modify `configs/experiment.yaml`:

```yaml
# Experiment identification
experiment_name: "german_lip_experiment"
description: "German authorship verification with linguistic features prompt"

# Dataset configuration
dataset: "amazon_review"  # Options: "amazon_review", "wikipedia"
dataset_language: "german"  # Options: "english", "german"

# Prompt configuration  
prompt: "lip"  # Options: "biased", "lip"
prompt_language: "german"  # Options: "english", "german"

# Sampling configuration
max_samples: 500
random_seed: 42

# Model configuration
model: "openai/gpt-4.1-mini"
# Available models:
# - "deepseek/deepseek-r1"
# - "openai/gpt-4.1" 
# - "openai/gpt-4o"
# - "openai/gpt-4o-mini"
# - "openai/gpt-4.1-mini"
# - "deepseek/deepseek-v3"
# - "openai/o3-mini"
# - "openai/gpt-4.1-nano"

# Optional: Custom data loading function
data_loader: null  # Options: null, "longest", "shortest", "longest_balanced", "shortest_balanced"

# Metrics configuration - specify which metrics to calculate and save
metrics:
  accuracy: true
  precision: true
  recall: true
  f1_score: true
  specificity: true
  npv: true  # Negative Predictive Value
  tp: true   # True Positive
  tn: true   # True Negative
  fp: true   # False Positive
  fn: true   # False Negative
  confusion_matrix: true
  token_usage: true
  costs: true

# Output configuration
output_dir: "results"
save_responses: true
save_metrics: true
save_costs: true
save_csv: true
```

### Advanced Usage

#### Run Multiple Experiments
```bash
python authorship_verification/batch_runner.py
```

#### Human Evaluation
```bash
python authorship_verification/human_evaluation.py --config configs/examples/human_evaluation.yaml
```

#### Analyze Results
```bash
# Analyze human evaluation results
python authorship_verification/human_evaluation.py --analyze results/human_evaluation/human_evaluation_german.csv

# Use Jupyter notebook for detailed analysis
jupyter notebook notebooks/results_analysis.ipynb
```

---

## 📊 Results

### Cross-lingual Performance Comparison

| Model | German F1 | English F1 | Δ (Eng-Ger) |
|-------|-----------|------------|-------------|
| GPT-4o | 0.742 | 0.785 | +0.043 |
| GPT-4.1 | 0.718 | 0.763 | +0.045 |
| GPT-4.1-mini | 0.695 | 0.731 | +0.036 |
| DeepSeek-R1 | 0.673 | 0.708 | +0.035 |
| DeepSeek-V3 | 0.681 | 0.715 | +0.034 |

*Results on Amazon review dataset with 500 samples, LIP prompting strategy*

### Prompt Strategy Comparison

| Strategy | German Accuracy | English Accuracy |
|----------|----------------|------------------|
| Biased | 0.672 | 0.695 |
| LIP | 0.718 | 0.742 |

### Human Baseline

- **German**: F1 = 0.823, Accuracy = 0.791
- **English**: F1 = 0.847, Accuracy = 0.815

---

## 📁 Project Structure

```
AV/
├── configs/                    # Configuration files
│   ├── experiment.yaml        # Main configuration template
│   └── examples/              # Example configurations
│       ├── german_lip.yaml
│       ├── english_biased.yaml
│       ├── longest_balanced.yaml
│       └── human_evaluation.yaml
├── authorship_verification/    # Main source code
│   ├── authorship_verification.py  # Main experiment runner
│   ├── batch_runner.py        # Batch experiment runner
│   ├── human_evaluation.py    # Human evaluation GUI
│   └── assets/                # Images and media files
│       └── Authorship_Verification_Processing.png
├── data/                      # Data loading utilities
│   └── data_loaders.py       # Dataset loading functions
├── results/                   # Results directory (auto-generated)
│   └── {language}_{prompt}_{data_loader}_{model}/
├── notebooks/                 # Analysis notebooks
│   └── results_analysis.ipynb
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── test_setup.py           # Setup verification script
└── README.md              # This documentation
```

---

## 🔧 Development

### Running Tests

```bash
# Verify setup
python test_setup.py

# Run with dependencies installed
pip install -r requirements.txt
python test_setup.py
```

### Adding New Models

1. Add model configuration in `authorship_verification.py`:
```python
self.model_costs = {
    "your/new-model": (input_cost_per_million, output_cost_per_million),
    # ...
}
```

2. Update the configuration examples and documentation.

### Adding New Datasets

1. Implement data loading function in `data/data_loaders.py`
2. Update configuration validation
3. Add example configuration

---

## 📝 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{al-labani-etal-2024-german-av,
  title={Are Large Language Models capable of Authorship Verification on German Texts?},
  author={Al-Labani, Abdullah and Devrani, Nitish and Leonard, Timothy},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2024},
  publisher={Association for Computational Linguistics},
  url={https://your-paper-url-here}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Issues and Questions

- 🐛 **Bug reports**: Please use the [issue tracker](../../issues)
- 💡 **Feature requests**: Open an issue with the `enhancement` label
- ❓ **Questions**: Use the [discussions](../../discussions) section

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Amazon Review Dataset**: Thanks to the creators of the multilingual authorship verification dataset
- **OpenAI & DeepSeek**: For providing access to their language models
- **Community**: All contributors and researchers working on authorship verification

---

<div align="center">
  <p>⭐ If you find this work useful, please consider giving us a star!</p>
</div>