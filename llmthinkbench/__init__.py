"""
LLMThinkBench: A framework for evaluating reasoning capabilities of Large Language Models.

This package provides tools to assess LLM performance on various reasoning tasks,
measuring accuracy, instruction following, and other metrics.

Features:
- 14+ mathematical and logical reasoning tasks
- Support for vLLM and Transformers backends with automatic fallback
- Multi-GPU tensor parallelism support
- Comprehensive evaluation metrics and reporting
- Professional CLI with detailed help and configuration options
"""

__version__ = "0.1.5"
__author__ = "Gaurav Srivastava"

from .tasks.base_task import BaseTask
from .tasks.sorting_task import SortingTask
from .tasks.comparison_task import ComparisonTask
from .tasks.sum_task import SumTask
from .tasks.multiplication_task import MultiplicationTask
from .tasks.odd_count_task import OddCountTask
from .tasks.even_count_task import EvenCountTask
from .tasks.absolute_difference_task import AbsoluteDifferenceTask
from .tasks.division_task import DivisionTask
from .tasks.find_maximum_task import FindMaximumTask
from .tasks.find_minimum_task import FindMinimumTask
from .tasks.mean_task import MeanTask
from .tasks.median_task import MedianTask
from .tasks.mode_task import ModeTask
from .tasks.subtraction_task import SubtractionTask
from .models.model_handler import ModelHandler
from .utils.reporting import generate_final_report

# Convenience function to run evaluations
def evaluate(model_id, tasks=None, **kwargs):
    """
    Evaluate an LLM on reasoning tasks.
    
    Args:
        model_id (str): Hugging Face model ID
        tasks (list): List of tasks to evaluate (default: all available tasks)
        **kwargs: Additional arguments for evaluation
            - datapoints (int): Number of samples per test case (default: 1000)
            - folds (int): Number of evaluation folds (default: 1)
            - range (list): Number range [min, max] (default: [-100, 100])
            - list_sizes (list): List sizes for list-based tasks (default: [8])
            - store_details (bool): Store detailed results (default: False)
            - output_dir (str): Output directory (default: auto-generated)
            - cuda_device (str): CUDA device (default: "cuda:0")
            - tensor_parallel_size (int): Number of GPUs (default: 1)
            - gpu_memory_utilization (float): GPU memory usage (default: 0.9)
            - temperature (float): Sampling temperature (default: 0.7)
            - top_p (float): Top-p sampling (default: 0.9)
            - max_tokens (int): Max generation tokens (default: 512)
            - trust_remote_code (bool): Trust remote code (default: False)
            - seed (int): Random seed (default: None)
        
    Returns:
        dict: Evaluation results
        
    Example:
        >>> import llmthinkbench
        >>> results = llmthinkbench.evaluate(
        ...     "Qwen/Qwen2.5-3B-Instruct", 
        ...     tasks=["sorting", "sum", "comparison"],
        ...     datapoints=500,
        ...     temperature=0.3
        ... )
    """
    from .cli import run_evaluation
    if tasks is None:
        tasks = [
            "sorting", "comparison", "sum", "multiplication", "odd_count", 
            "even_count", "absolute_difference", "division", "find_maximum", 
            "find_minimum", "mean", "median", "mode", "subtraction"
        ]
    return run_evaluation(model_id, tasks, **kwargs)