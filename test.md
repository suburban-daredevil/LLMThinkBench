# 🧠 LLMThinkBench

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.6-blue)](https://pypi.org/project/llmthinkbench/0.1.6/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![vLLM](https://img.shields.io/badge/Powered%20by-vLLM-orange)](https://github.com/vllm-project/vllm)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/)

## A Framework for Evaluating Basic Math Reasoning Capabilities and Overthinking of Language Models

LLMThinkBench is a comprehensive framework designed to rigorously evaluate the basic math reasoning capabilities of Language Models, while also identifying instances of overthinking—where models apply unnecessarily complex logic to simple problems. Through standardized and reproducible benchmarks, it offers valuable insights into how well models perform on various reasoning tasks, from basic arithmetic to complex logical operations.

<!-- <div align="center">
  <img src="https://raw.githubusercontent.com/ctrl-gaurav/LLMThinkBench/main/assets/llmthinkbench_banner.png" alt="LLMThinkBench" width="600"/>
</div> -->

<!-- <div align="center">
  <span style="font-size: 240px;">🧠</span>
</div> -->

## 📰 News & Releases

- **v0.1.6** (Latest) - **Revolutionary Overthinking Metrics**: Introduced advanced **Efficiency Score** and **Overthinking Ratio** metrics that properly balance accuracy and verbosity. Enhanced reporting with efficiency rankings to identify models that achieve high accuracy with minimal tokens.
- **v0.1.5** - Major backend revamp with robust error handling, intelligent fallback support, better token estimation, flexible device control, and smoother interruption handling.
- **v0.1.4** - Major improvements to parsing robustness for fair evaluations. Enhanced result validation mechanisms and edge case handling.
- **v0.1.3** - Added mean, median, mode tasks and implemented GPU customization options, allowing users to specify GPU memory allocation.
- **v0.1.2** - Expanded task library with find_maximum, find_minimum, absolute_difference, and division tasks. Improved documentation.
- **v0.1.1** - Fixed several inference issues and optimized performance using vLLM for high-throughput evaluation.
- **v0.1.0** - Initial release with core functionality including sorting and comparison tasks.

## 🌟 Key Features

- **Comprehensive Evaluation**: Test LLMs on a range of mathematical and logical reasoning tasks
- **🎯 Advanced Overthinking Detection**: Novel efficiency metrics that balance accuracy and conciseness
- **📊 Efficiency Rankings**: Identify models that achieve high performance without excessive verbosity
- **Modular Architecture**: Easily extend with custom evaluation tasks
- **Fair Comparison**: Standardized methodology for comparing models
- **Efficient Inference**: Built on vLLM for high-throughput batched evaluation
- **Detailed Metrics**: Comprehensive reports on accuracy, instruction following, and output characteristics
- **Multi-GPU Support**: Scale evaluations across multiple GPUs
- **Reproducible Results**: Consistent methodology across model comparisons
- **Output Analysis**: Identify when and how models make reasoning errors

## 🧮 Revolutionary Overthinking Metrics

### Efficiency Score
**Formula**: `Efficiency Score = 2 × (Accuracy × Token_Efficiency) / (Accuracy + Token_Efficiency)`

Where:
- `Token_Efficiency = 1 - normalized_tokens`
- `normalized_tokens = (tokens - min_tokens) / (max_tokens - min_tokens)`

**Benefits**:
- **Balances accuracy and conciseness** using harmonic mean
- **Penalizes verbosity** while rewarding high accuracy
- **Provides intuitive ranking**: Higher score = better model
- **Handles edge cases** gracefully

### Overthinking Ratio
**Formula**: `Overthinking Ratio = (tokens - baseline) / accuracy`

**Interpretation**:
- **Lower values indicate less overthinking**
- Measures "extra tokens per accuracy point"
- Helps identify models that are unnecessarily verbose
- Infinite ratio for models with zero accuracy

### Example Comparison
| Model | Accuracy | Tokens | Efficiency Score | Overthinking Ratio |
|-------|----------|--------|------------------|-------------------|
| Model A | 98% | 100 | 0.892 | 50.0 |
| Model B | 99% | 120 | 0.881 | 70.7 |

**Result**: Model A is more efficient despite slightly lower accuracy!

## 📊 Supported Tasks

| Task Type | Task | Description |
|-----------|------|-------------|
| **Basic Operations** | Sorting | Evaluates ability to correctly sort lists of numbers |
| | Comparison | Tests number comparison abilities (greater than, less than, equal to) |
| | Sum | Assesses ability to calculate the sum of multiple numbers |
| | Subtraction | Measures accuracy in subtracting two numbers |
| | Multiplication | Tests multiplication of numbers |
| | Division | Evaluates division operations |
| **List Processing** | Find Maximum | Finds the largest value in a list |
| | Find Minimum | Identifies the smallest value in a list |
| | Odd Count | Counts odd numbers in a list |
| | Even Count | Counts even numbers in a list |
| **Statistical** | Mean | Calculates the arithmetic mean of a list |
| | Median | Finds the median value of a list |
| | Mode | Identifies the most frequent value(s) in a list |
| **Advanced** | Absolute Difference | Calculates the absolute difference between numbers |

## 🚀 Installation

```bash
# Install from PyPI
pip install llmthinkbench

# Install from source
git clone https://github.com/ctrl-gaurav/llmthinkbench.git
cd llmthinkbench
pip install -e .
```

## 📈 Quick Start

### Command Line Interface

```bash
# Basic usage
llmthinkbench --model_id "Qwen/Qwen2.5-1.5B-Instruct" --tasks sorting comparison

# Comprehensive evaluation
llmthinkbench --model_id "meta-llama/Llama-2-7b-chat-hf" \
  --tensor_parallel_size 2 \
  --tasks sorting comparison sum multiplication \
  --datapoints 1000 \
  --list_sizes 8 16 32 \
  --folds 3 \
  --range -1000 1000 \
  --store_details \
  --output_dir "./llama2_evaluation_results"
```

### Python API

```python
from llmthinkbench import evaluate

# Simple evaluation
results = evaluate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    tasks=["sorting", "comparison", "sum"]
)

# Advanced configuration
results = evaluate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    tasks=["sorting", "comparison", "sum", "multiplication"],
    datapoints=500,
    list_sizes=[8, 16, 32],
    folds=3,
    range=[-1000, 1000],
    store_details=True,
    output_dir="./custom_results",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)
```

### Detailed API Usage

```python
from llmthinkbench.models.model_handler import ModelHandler
from llmthinkbench.tasks.sorting_task import SortingTask
from llmthinkbench.tasks.comparison_task import ComparisonTask
from llmthinkbench.utils.reporting import generate_final_report

# Initialize model
model_handler = ModelHandler(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9
)

# Configure output directory
output_dir = "llama2_eval_results"

# Run sorting task
sorting = SortingTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-100,
    max_val=100,
    num_folds=3,
    num_samples=500,
    store_details=True,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Evaluate multiple list sizes
list_sizes = [8, 16, 32]
sorting_metrics = sorting.run_evaluation(list_sizes)

# Run comparison task
comparison = ComparisonTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-100,
    max_val=100,
    num_folds=3,
    num_samples=500,
    store_details=True,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

comparison_metrics = comparison.run_evaluation()

# Generate comprehensive report with efficiency metrics
all_metrics = sorting_metrics + comparison_metrics
report = generate_final_report(all_metrics, list_sizes, output_dir)
```

## 📊 Example Results with New Metrics

Below is an example report generated by LLMThinkBench v0.1.6:

```
+------------------+----------------------------+------------------+---------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+
| Task             | Accuracy                   | Efficiency Score | Overthinking Ratio  | Instruction Followed              | Tokens                  | Chars                        | Words                        |
+------------------+----------------------------+------------------+---------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+
| sorting_8        | 95.20% ± 3.60              | 0.892            | 50.0                | 98.80% ± 1.20                     | 186.2 ± 32.6            | 612.6 ± 98.4                 | 93.5 ± 15.6                  |
| sorting_16       | 87.40% ± 4.80              | 0.743            | 89.2                | 96.70% ± 2.30                     | 312.5 ± 48.7            | 982.3 ± 156.5                | 167.9 ± 26.9                 |
| sorting_32       | 68.60% ± 7.20              | 0.521            | 198.5               | 92.40% ± 3.50                     | 645.7 ± 92.2            | 1872.2 ± 283.6               | 348.8 ± 52.8                 |
| comparison       | 99.20% ± 1.20              | 0.951            | 44.1                | 99.60% ± 0.50                     | 93.8 ± 16.2             | 324.8 ± 52.2                 | 48.3 ± 8.1                   |
| sum_8            | 97.80% ± 2.10              | 0.923            | 35.2                | 99.30% ± 0.70                     | 134.6 ± 23.9            | 452.2 ± 78.3                 | 68.9 ± 11.7                  |
| multiplication   | 94.60% ± 3.50              | 0.885            | 47.9                | 98.40% ± 1.60                     | 114.3 ± 19.4            | 386.7 ± 64.3                 | 58.4 ± 9.7                   |
+------------------+----------------------------+------------------+---------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+

🏆 Efficiency Rankings (Best Balance of Accuracy & Conciseness)
+--------+---------------+------------------+----------+--------+---------------------+
|  Rank  | Task          | Efficiency Score | Accuracy | Tokens | Overthinking Ratio  |
+--------+---------------+------------------+----------+--------+---------------------+
|   1    | comparison    | 0.951            | 99.2%    | 93.8   | 44.1                |
|   2    | sum_8         | 0.923            | 97.8%    | 134.6  | 35.2                |
|   3    | sorting_8     | 0.892            | 95.2%    | 186.2  | 50.0                |
|   4    | multiplication| 0.885            | 94.6%    | 114.3  | 47.9                |
|   5    | sorting_16    | 0.743            | 87.4%    | 312.5  | 89.2                |
|   6    | sorting_32    | 0.521            | 68.6%    | 645.7  | 198.5               |
+--------+---------------+------------------+----------+--------+---------------------+
```

## 📈 Visualization with New Metrics

You can visualize LLMThinkBench results including the new efficiency metrics:

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load results
with open("final_report.json") as f:
    results = json.load(f)

# Extract efficiency rankings
rankings = results['efficiency_rankings']

# Create dataframe for plotting
df = pd.DataFrame(rankings)

# Plot efficiency comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Efficiency Score vs Accuracy
scatter = ax1.scatter(df['accuracy']*100, df['efficiency_score'], 
                     s=100, alpha=0.7, c=df['tokens'], cmap='viridis')
ax1.set_xlabel('Accuracy (%)')
ax1.set_ylabel('Efficiency Score')
ax1.set_title('Efficiency Score vs Accuracy')
plt.colorbar(scatter, ax=ax1, label='Tokens')

# Add task labels
for i, row in df.iterrows():
    ax1.annotate(row['task'], (row['accuracy']*100, row['efficiency_score']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# Efficiency Score ranking
ax2.barh(range(len(df)), df['efficiency_score'])
ax2.set_yticks(range(len(df)))
ax2.set_yticklabels(df['task'])
ax2.set_xlabel('Efficiency Score')
ax2.set_title('Efficiency Score Rankings')
ax2.invert_yaxis()

# Overthinking Ratio vs Accuracy
# Filter out infinite values for plotting
finite_df = df[df['overthinking_ratio'] != float('inf')]
scatter2 = ax3.scatter(finite_df['accuracy']*100, finite_df['overthinking_ratio'], 
                      s=100, alpha=0.7, c=finite_df['tokens'], cmap='plasma')
ax3.set_xlabel('Accuracy (%)')
ax3.set_ylabel('Overthinking Ratio')
ax3.set_title('Overthinking Ratio vs Accuracy')
plt.colorbar(scatter2, ax=ax3, label='Tokens')

# Tokens vs Accuracy with Efficiency Score as color
scatter3 = ax4.scatter(df['accuracy']*100, df['tokens'], 
                      s=100, alpha=0.7, c=df['efficiency_score'], cmap='coolwarm')
ax4.set_xlabel('Accuracy (%)')
ax4.set_ylabel('Tokens')
ax4.set_title('Tokens vs Accuracy (colored by Efficiency)')
plt.colorbar(scatter3, ax=ax4, label='Efficiency Score')

plt.tight_layout()
plt.savefig("efficiency_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
```

## ⚙️ Advanced Configuration

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_id` | Hugging Face model ID | *Required* |
| `--tasks` | Tasks to evaluate | `["sorting"]` |
| `--datapoints` | Number of samples per test case | `1000` |
| `--