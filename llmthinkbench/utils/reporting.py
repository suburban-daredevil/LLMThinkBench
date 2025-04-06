import os
import json
import logging
import numpy as np
from tabulate import tabulate

def format_report_table(report_data):
    """Create pretty-printed table from report data"""
    headers = [
        "Test Case", "Accuracy (Mean)", "Accuracy (Std)", 
        "Instruction Followed", "Avg Chars", "Avg Words", "Avg Tokens"
    ]
    
    rows = []
    for size, metrics in report_data.items():
        rows.append([
            size,
            f"{metrics['accuracy']['mean']*100:.2f}%",
            f"{metrics['accuracy']['std']*100:.2f}%" if metrics['accuracy']['std'] else "-",
            f"{metrics['instruction_followed']['mean']*100:.2f}%",
            metrics['avg_response_length'],
            metrics['avg_word_count'],
            metrics['avg_output_tokens']
        ])
    
    return tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f")

def generate_final_report(all_metrics, list_sizes, output_dir):
    """Generate aggregated report across all test cases with rounding"""
    report = {}
    
    # Group metrics by task
    tasks = {}
    for metric in all_metrics:
        task_name = metric.get('task', 'unknown')
        if task_name not in tasks:
            tasks[task_name] = []
        tasks[task_name].append(metric)
    
    # Process metrics for each task
    for task_name, task_metrics in tasks.items():
        if task_name == 'sorting':
            # Process sorting task by list size
            for size in list_sizes:
                size_metrics = [m for m in task_metrics if m.get('list_size') == size]
                if not size_metrics:
                    continue
                    
                report[f"sorting_{size}"] = {
                    'accuracy': {
                        'mean': round(np.mean([m['accuracy'] for m in size_metrics]), 4),
                        'std': round(np.std([m['accuracy'] for m in size_metrics]), 4) if len(size_metrics) > 1 else 0
                    },
                    'instruction_followed': {
                        'mean': round(np.mean([m['instruction_followed_pct'] for m in size_metrics]), 4),
                        'std': round(np.std([m['instruction_followed_pct'] for m in size_metrics]), 4) if len(size_metrics) > 1 else 0
                    },
                    'avg_response_length': round(np.mean([m['avg_response_length'] for m in size_metrics]), 2),
                    'avg_word_count': round(np.mean([m['avg_word_count'] for m in size_metrics]), 2),
                    'avg_output_tokens': round(np.mean([m['avg_output_tokens'] for m in size_metrics]), 2)
                }
        else:
            # Process other tasks without list size
            report[task_name] = {
                'accuracy': {
                    'mean': round(np.mean([m['accuracy'] for m in task_metrics]), 4),
                    'std': round(np.std([m['accuracy'] for m in task_metrics]), 4) if len(task_metrics) > 1 else 0
                },
                'instruction_followed': {
                    'mean': round(np.mean([m['instruction_followed_pct'] for m in task_metrics]), 4),
                    'std': round(np.std([m['instruction_followed_pct'] for m in task_metrics]), 4) if len(task_metrics) > 1 else 0
                },
                'avg_response_length': round(np.mean([m['avg_response_length'] for m in task_metrics]), 2),
                'avg_word_count': round(np.mean([m['avg_word_count'] for m in task_metrics]), 2),
                'avg_output_tokens': round(np.mean([m['avg_output_tokens'] for m in task_metrics]), 2)
            }
    
    # Save final report
    report_file = os.path.join(output_dir, "final_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create and save formatted table
    table = format_report_table(report)
    table_file = os.path.join(output_dir, "results_table.txt")
    with open(table_file, 'w') as f:
        f.write(table)
    
    logging.info(f"\nResults Summary:\n{table}")
    logging.info(f"Saved final report to {report_file}")
    return report