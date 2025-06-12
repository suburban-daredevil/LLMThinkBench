#!/usr/bin/env python3
"""
LLMThinkBench CLI - Command Line Interface for LLM Reasoning Evaluation
"""

import sys
import os

# Handle version check first (before any heavy imports)
if len(sys.argv) > 1 and '--version' in sys.argv:
    # Quick version check without importing the entire package
    try:
        from llmthinkbench import __version__
        print(f"LLMThinkBench version {__version__}")
        sys.exit(0)
    except ImportError:
        print("LLMThinkBench version 0.1.5")
        sys.exit(0)

# Handle CUDA device selection early (before any CUDA-related imports)

def setup_cuda_device():
    """Setup CUDA device(s) based on command line arguments"""
    cuda_device = None
    for i, arg in enumerate(sys.argv):
        if arg == '--cuda_device' and i + 1 < len(sys.argv):
            cuda_device = sys.argv[i + 1]
            break

    if cuda_device:
        # Remove whitespace and split by comma
        devices = [d.strip() for d in cuda_device.split(',') if d.strip()]
        if all(d.startswith('cuda:') for d in devices):
            device_ids = [d.split(':')[1] for d in devices]
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids)
        else:
            raise ValueError(f"❌ Invalid CUDA device format: {cuda_device}. Expected format: cuda:0, cuda:1, etc.")



# def setup_cuda_device():
#     """Setup CUDA device based on command line arguments"""
#     cuda_device = None
#     for i, arg in enumerate(sys.argv):
#         if arg == '--cuda_device' and i + 1 < len(sys.argv):
#             cuda_device = sys.argv[i + 1]
#             break
    
#     if cuda_device:
#         # Validate CUDA device format
#         if cuda_device.startswith('cuda:'):
#             device_id = cuda_device.split(':')[1]
#             if device_id.isdigit():
#                 os.environ['CUDA_VISIBLE_DEVICES'] = device_id
#                 print(f"🔧 CUDA device set to: {cuda_device}")
#             else:
#                 print(f"❌ Invalid CUDA device format: {cuda_device}. Expected format: cuda:0, cuda:1, etc.")
#                 sys.exit(1)
#         else:
#             print(f"❌ Invalid CUDA device format: {cuda_device}. Expected format: cuda:0, cuda:1, etc.")
#             sys.exit(1)
#     else:
#         # Default to cuda:0
#         os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Setup CUDA device early
setup_cuda_device()

# Now import the rest of the modules
import argparse
import logging
import json
import random
from datetime import datetime
import importlib

from llmthinkbench.utils.logging_utils import setup_logging
from llmthinkbench.utils.reporting import generate_final_report
from llmthinkbench.models.model_handler import ModelHandler
from llmthinkbench.tasks.sorting_task import SortingTask
from llmthinkbench.tasks.comparison_task import ComparisonTask
from llmthinkbench.tasks.sum_task import SumTask
from llmthinkbench.tasks.multiplication_task import MultiplicationTask
from llmthinkbench.tasks.odd_count_task import OddCountTask
from llmthinkbench.tasks.even_count_task import EvenCountTask
from llmthinkbench.tasks.absolute_difference_task import AbsoluteDifferenceTask
from llmthinkbench.tasks.division_task import DivisionTask
from llmthinkbench.tasks.find_maximum_task import FindMaximumTask
from llmthinkbench.tasks.find_minimum_task import FindMinimumTask
from llmthinkbench.tasks.mean_task import MeanTask
from llmthinkbench.tasks.median_task import MedianTask
from llmthinkbench.tasks.mode_task import ModeTask
from llmthinkbench.tasks.subtraction_task import SubtractionTask

def show_intro():
    """Display professional introduction banner"""
    try:
        from llmthinkbench import __version__
        version = __version__
    except ImportError:
        version = "5"
    
    intro = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██╗     ██╗     ███╗   ███╗████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗          ║
║    ██║     ██║     ████╗ ████║╚══██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝          ║
║    ██║     ██║     ██╔████╔██║   ██║   ███████║██║██╔██╗ ██║█████╔╝           ║
║    ██║     ██║     ██║╚██╔╝██║   ██║   ██╔══██║██║██║╚██╗██║██╔═██╗           ║
║    ███████╗███████╗██║ ╚═╝ ██║   ██║   ██║  ██║██║██║ ╚████║██║  ██╗          ║
║    ╚══════╝╚══════╝╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝          ║
║                                                                               ║
║                        ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗             ║
║                        ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║             ║
║                        ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║             ║
║                        ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║             ║
║                        ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║             ║
║                        ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝             ║
║                                                                               ║
║       🧠 Advanced LLM Reasoning and Overthinking Evaluation Framework         ║
║                        📊   Version {version:<15}                           ║
║                                                                               ║
║   A comprehensive framework for evaluating Large Language Model reasoning     ║
║   and overthinking capabilities across quantitative and logical tasks with    ║
║   robust model support including vLLM and Transformers backends.              ║
║                                                                               ║
║   🚀 Features:                                                                ║
║     • 14+ Mathematical & Logical Reasoning Tasks                              ║
║     • Support for vLLM & Transformers backends                                ║
║     • Automatic fallback from vLLM to Transformers                            ║ 
║     • Multi-GPU tensor parallelism and flexible device mapping                ║
║     • Comprehensive task suite for reasoning evaluation                       ║
║     • Detailed reporting & analytics                                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(intro)

def parse_arguments():
    """Parse command line arguments with detailed help"""
    parser = argparse.ArgumentParser(
        description='🧠 LLMThinkBench: Comprehensive evaluation framework for Large Language Model reasoning capabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 Examples:
  Basic evaluation:
    llmthinkbench --model_id microsoft/DialoGPT-medium --tasks sorting sum

  Advanced configuration:
    llmthinkbench --model_id Qwen/Qwen2.5-3B-Instruct --tasks sorting comparison sum \\
                  --datapoints 500 --folds 3 --cuda_device cuda:1 --temperature 0.5 \\
                  --store_details --trust_remote_code

  Multi-task evaluation:
    llmthinkbench --model_id your/model --tasks sorting sum multiplication division \\
                  --list_sizes 4 8 16 --range -50 50 --max_tokens 1024

🔗 For more information, visit: https://github.com/yourusername/llmthinkbench
        """
    )
    
    # Version (handled early, but keep for help)
    parser.add_argument('--version', action='version', 
                        version=f'LLMThinkBench {importlib.import_module("llmthinkbench").__version__}',
                        help='Show version information and exit')
    
    # Core Configuration
    core_group = parser.add_argument_group('🎯 Core Configuration', 'Essential parameters for evaluation')
    core_group.add_argument('--model_id', required=True, 
                           help='🤖 Hugging Face model identifier (e.g., "microsoft/DialoGPT-medium", "Qwen/Qwen2.5-3B-Instruct")')
    
    core_group.add_argument('--tasks', nargs='+', default=['sorting'], 
                           choices=['sorting', 'comparison', 'sum', 'multiplication', 'odd_count', 
                                   'even_count', 'absolute_difference', 'division', 'find_maximum', 
                                   'find_minimum', 'mean', 'median', 'mode', 'subtraction'],
                           help='📝 List of reasoning tasks to evaluate. Available tasks: sorting (arrange numbers), '
                                'comparison (compare two numbers), sum (add list of numbers), multiplication (multiply numbers), '
                                'odd_count (count odd numbers), even_count (count even numbers), absolute_difference (|a-b|), '
                                'division (a÷b), find_maximum (max value), find_minimum (min value), mean (average), '
                                'median (middle value), mode (most frequent), subtraction (a-b)')
    
    # Evaluation Parameters
    eval_group = parser.add_argument_group('📊 Evaluation Parameters', 'Control the scope and depth of evaluation')
    eval_group.add_argument('--datapoints', type=int, default=1000, 
                           help='🔢 Number of test samples to generate per task configuration (default: 1000). '
                                'Higher values provide more robust statistics but take longer to run')
    
    eval_group.add_argument('--folds', type=int, default=1, 
                           help='🔄 Number of evaluation folds for cross-validation (default: 1). '
                                'Multiple folds help assess consistency of model performance')
    
    eval_group.add_argument('--range', nargs=2, type=int, default=[-100, 100], 
                           help='📏 Numerical range for generating test numbers (default: -100 100). '
                                'Format: --range MIN_VALUE MAX_VALUE')
    
    eval_group.add_argument('--list_sizes', nargs='+', type=int, default=[8], 
                           help='📋 List lengths to test for list-based tasks (default: [8]). '
                                'Example: --list_sizes 4 8 16 tests performance on different complexity levels')
    
    eval_group.add_argument('--store_details', action='store_true', 
                           help='💾 Save detailed per-example results including raw model responses, '
                                'parsed answers, and correctness for deep analysis')
    
    eval_group.add_argument('--output_dir', type=str, default=None,
                           help='📁 Custom directory path for saving results (default: auto-generated based on model and timestamp)')
    
    # Hardware Configuration
    hardware_group = parser.add_argument_group('⚙️ Hardware Configuration', 'GPU and device management')
    hardware_group.add_argument('--cuda_device', type=str, default='cuda:0',
                               help='🎮 CUDA device to use (default: cuda:0). Format: cuda:0, cuda:1, etc. '
                                    'Automatically sets CUDA_VISIBLE_DEVICES environment variable')
    
    hardware_group.add_argument('--tensor_parallel_size', type=int, default=1, 
                               help='🔗 Number of GPUs for tensor parallelism (default: 1). '
                                    'Use >1 for large models that don\'t fit on single GPU')
    
    hardware_group.add_argument('--gpu_memory_utilization', type=float, default=0.9, 
                               help='🧠 GPU memory utilization ratio (default: 0.9). '
                                    'Lower values (0.7-0.8) if experiencing OOM errors')
    
    # Model Behavior
    model_group = parser.add_argument_group('🤖 Model Behavior', 'Parameters controlling model inference')
    model_group.add_argument('--temperature', type=float, default=0.7, 
                            help='🌡️ Sampling temperature for randomness control (default: 0.7). '
                                 'Lower values (0.1-0.3) for more deterministic outputs, higher (0.8-1.0) for creativity')
    
    model_group.add_argument('--top_p', type=float, default=0.9, 
                            help='🎯 Nucleus sampling top-p value (default: 0.9). '
                                 'Controls diversity by sampling from top p% probability mass')
    
    model_group.add_argument('--max_tokens', type=int, default=512, 
                            help='📏 Maximum tokens for model generation (default: 512). '
                                 'Increase for tasks requiring longer explanations')
    
    model_group.add_argument('--trust_remote_code', action='store_true',
                            help='🔓 Allow execution of remote code for custom models (default: False). '
                                 'Only enable for trusted models - security risk otherwise')
    
    # Reproducibility
    repro_group = parser.add_argument_group('🎲 Reproducibility', 'Ensure consistent results across runs')
    repro_group.add_argument('--seed', type=int, default=None,
                            help='🌱 Random seed for reproducible results (default: random). '
                                 'Use same seed to get identical results across runs')

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
        'comparison': ComparisonTask,
        'sum': SumTask,
        'multiplication': MultiplicationTask,
        'odd_count': OddCountTask,
        'even_count': EvenCountTask,
        'absolute_difference': AbsoluteDifferenceTask,
        'division': DivisionTask,
        'find_maximum': FindMaximumTask,
        'find_minimum': FindMinimumTask,
        'mean': MeanTask,
        'median': MedianTask,
        'mode': ModeTask,
        'subtraction': SubtractionTask
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
        logging.error(f"❌ Failed to load task '{task_name}': {e}")
        sys.exit(1)

def run_evaluation(model_id, tasks, **kwargs):
    """Run evaluation programmatically"""
    # This function is used by the convenience function in __init__.py
    # Convert kwargs to args-like object
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Set defaults
    args = Args(
        model_id=model_id,
        tasks=tasks,
        datapoints=kwargs.get('datapoints', 1000),
        folds=kwargs.get('folds', 1),
        range=kwargs.get('range', [-100, 100]),
        list_sizes=kwargs.get('list_sizes', [8]),
        store_details=kwargs.get('store_details', False),
        output_dir=kwargs.get('output_dir', None),
        cuda_device=kwargs.get('cuda_device', 'cuda:0'),
        tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
        gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
        temperature=kwargs.get('temperature', 0.7),
        top_p=kwargs.get('top_p', 0.9),
        max_tokens=kwargs.get('max_tokens', 512),
        trust_remote_code=kwargs.get('trust_remote_code', False),
        seed=kwargs.get('seed', None)
    )
    
    return main_evaluation(args)

def main_evaluation(args):
    """Main evaluation logic (separated for reuse)"""
    # Create output directory
    output_dir = create_output_directory(args)
    
    # Set up logging
    log_file = setup_logging(output_dir)
    logging.info(f"🚀 Starting LLMThinkBench evaluation with parameters:\n{json.dumps(vars(args), indent=2)}")

    # Initialize model handler
    model_handler = ModelHandler(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code
    )
    
    all_metrics = []
    
    # Run each requested task
    for task_name in args.tasks:
        logging.info(f"\n{'='*40}\n🎯 Running task: {task_name}\n{'='*40}")
        
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
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        
        # Task-specific configuration
        if task_name in ['sorting', 'sum', 'odd_count', 'even_count', 'find_maximum', 'find_minimum', 'mean', 'median', 'mode']:
            task_metrics = task.run_evaluation(args.list_sizes)
        elif task_name == 'multiplication':
            # Use specific list sizes for multiplication task: 2, 4, and 8, error beyond list size 8
            task_metrics = task.run_evaluation([16])
        else:
            # Generic task interface or pair-based tasks
            task_metrics = task.run_evaluation()
            
        all_metrics.extend(task_metrics)
    
    # Generate final report if metrics were collected
    if all_metrics:
        final_report = generate_final_report(all_metrics, args.list_sizes, output_dir)
    
    logging.info(f"\n{'='*40}\n✅ LLMThinkBench evaluation complete")
    logging.info(f"📁 All results saved to: {output_dir}")
    logging.info(f"📋 Log file: {log_file}")
    
    return all_metrics

def main():
    # Show introduction
    show_intro()
    
    # Parse arguments
    args = parse_arguments()

    # Check if model_id is provided
    if not args.model_id:
        sys.exit(0)
    
    return main_evaluation(args)

if __name__ == "__main__":
    main()