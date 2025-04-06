# llmthinkbench: LLM Reasoning Evaluation Framework

A framework for evaluating overthinking and basic reasoning capabilities of Large Language Models

## Features

- Modular architecture for easy addition of new evaluation tasks
- Built-in tasks: sorting, number comparison
- Detailed reporting and metrics
- Efficient batched inference using vLLM

## Installation

```bash
pip install llmthinkbench
```

## Quick Start

```bash
# Run evaluation with default parameters
llmthinkbench --model_id "Qwen/Qwen2.5-1.5B-Instruct" --tasks sorting comparison

# Run with custom parameters
llmthinkbench --model_id "meta-llama/Llama-2-7b-chat-hf" \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.9 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_tokens 512 \
  --tasks sorting comparison \
  --datapoints 100 \
  --list_sizes 8 16 32 \
  --folds 3 \
  --range -100 100 \
  --store_details
```

## Adding New Tasks

1. Create a new task module in `llmthinkbench/tasks/your_task.py`
2. Implement a class that inherits from `BaseTask` and implements required methods
3. Register your task in `llmthinkbench/tasks/__init__.py`
4. Run with `--tasks your_task`

## License

MIT License