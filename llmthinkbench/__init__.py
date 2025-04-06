"""
LLMThinkBench: A framework for evaluating reasoning capabilities of Large Language Models.

This package provides tools to assess LLM performance on various reasoning tasks,
measuring accuracy, instruction following, and other metrics.
"""

__version__ = "0.1.0"
__author__ = "Gaurav Srivastava"

from .tasks.base_task import BaseTask
from .tasks.sorting_task import SortingTask
from .tasks.comparison_task import ComparisonTask
from .models.model_handler import ModelHandler
from .utils.reporting import generate_final_report

# Convenience function to run evaluations
def evaluate(model_id, tasks=None, **kwargs):
    """
    Evaluate an LLM on reasoning tasks.
    
    Args:
        model_id (str): Hugging Face model ID
        tasks (list): List of tasks to evaluate (default: ["sorting", "comparison"])
        **kwargs: Additional arguments for evaluation
        
    Returns:
        dict: Evaluation results
    """
    from .cli import run_evaluation
    if tasks is None:
        tasks = ["sorting", "comparison"]
    return run_evaluation(model_id, tasks, **kwargs)