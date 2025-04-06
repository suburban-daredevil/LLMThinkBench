import argparse
import os
import sys
import logging
import json
from datetime import datetime
import importlib

from llmthinkbench.utils.logging_utils import setup_logging
from llmthinkbench.utils.reporting import generate_final_report
from llmthinkbench.models.model_handler import ModelHandler
from llmthinkbench.tasks.sorting_task import SortingTask
from llmthinkbench.tasks.comparison_task import ComparisonTask

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LLMThinkBench: Evaluate LLM reasoning capabilities')
    parser.add_argument('--model_id', required=True, help='Hugging Face model ID')
    parser.add_argument('--tasks', nargs='+', default=['sorting'], 
                        help='Tasks to evaluate (e.g., sorting, comparison)')
    parser.add_argument('--datapoints', type=int, default=1000, 
                        help='Number of samples per test case')
    parser.add_argument('--folds', type=int, default=1, 
                        help='Number of evaluation folds')
    parser.add_argument('--range', nargs=2, type=int, default=[-100, 100], 
                        help='Number range for evaluation')
    parser.add_argument('--list_sizes', nargs='+', type=int, default=[8], 
                        help='List sizes to evaluate (for sorting task)')
    parser.add_argument('--store_details', action='store_true', 
                        help='Store detailed per-example results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: auto-generated)')
    # GPU configuration
    parser.add_argument('--tensor_parallel_size', type=int, default=1, 
                        help='Number of GPUs/CUDA devices to use')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, 
                        help='GPU memory utilization threshold')
    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.7, 
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, 
                        help='Sampling top_p value')
    parser.add_argument('--max_tokens', type=int, default=512, 
                        help='Maximum tokens for sampling')
    return parser.parse_args()

def create_output_directory(args):
    """Create output directory based on model name and tasks"""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = args.model_id.split('/')[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{model_name}_{'_'.join(args.tasks)}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_task_class(task_name):
    """Get task class based on task name"""
    task_mapping = {
        'sorting': SortingTask,
        'comparison': ComparisonTask
    }
    
    if task_name in task_mapping:
        return task_mapping[task_name]
    
    try:
        # Try to load a custom task dynamically
        module_path = f"llmthinkbench.tasks.{task_name}_task"
        module = importlib.import_module(module_path)
        class_name = ''.join(word.capitalize() for word in task_name.split('_')) + 'Task'
        task_class = getattr(module, class_name)
        return task_class
    except (ImportError, AttributeError) as e:
        logging.error(f"Failed to load task '{task_name}': {e}")
        sys.exit(1)

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = create_output_directory(args)
    
    # Set up logging
    log_file = setup_logging(output_dir)
    logging.info(f"Starting LLMThinkBench evaluation with parameters:\n{json.dumps(vars(args), indent=2)}")

    # Initialize model handler
    model_handler = ModelHandler(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    all_metrics = []
    
    # Run each requested task
    for task_name in args.tasks:
        logging.info(f"\n{'='*40}\nRunning task: {task_name}\n{'='*40}")
        
        # Get the task class
        task_class = load_task_class(task_name)
        
        # Initialize the task
        task = task_class(
            model_handler=model_handler,
            output_dir=output_dir,
            min_val=args.range[0],
            max_val=args.range[1],
            num_folds=args.folds,
            num_samples=args.datapoints,
            store_details=args.store_details,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        # Task-specific configuration
        if task_name == 'sorting':
            task_metrics = task.run_evaluation(args.list_sizes)
        elif task_name == 'comparison':
            task_metrics = task.run_evaluation()
        else:
            # Generic task interface
            task_metrics = task.run_evaluation()
            
        all_metrics.extend(task_metrics)
    
    # Generate final report if metrics were collected
    if all_metrics:
        final_report = generate_final_report(all_metrics, args.list_sizes, output_dir)
    
    logging.info(f"\n{'='*40}\nLLMThinkBench evaluation complete")
    logging.info(f"All results saved to: {output_dir}")
    logging.info(f"Log file: {log_file}")

if __name__ == "__main__":
    main()