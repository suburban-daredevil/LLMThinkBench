"""Task implementations for LLMThinkBench."""

from .base_task import BaseTask
from .sorting_task import SortingTask
from .comparison_task import ComparisonTask

# Dict mapping task names to task classes
AVAILABLE_TASKS = {
    "sorting": SortingTask,
    "comparison": ComparisonTask
}

def get_task_class(task_name):
    """Get task class by name."""
    if task_name not in AVAILABLE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(AVAILABLE_TASKS.keys())}")
    return AVAILABLE_TASKS[task_name]

__all__ = ["BaseTask", "SortingTask", "ComparisonTask", "get_task_class", "AVAILABLE_TASKS"]