import os
import json
import logging
import numpy as np
from tabulate import tabulate

def calculate_efficiency_score(accuracy, tokens, all_tokens):
    """
    Calculate efficiency score that balances accuracy and token efficiency.
    
    Efficiency Score = 2 × (Accuracy × Token_Efficiency) / (Accuracy + Token_Efficiency)
    
    Where:
    - Token_Efficiency = 1 - normalized_tokens
    - normalized_tokens = (tokens - min_tokens) / (max_tokens - min_tokens)
    
    Args:
        accuracy (float): Model accuracy (0-1)
        tokens (float): Average tokens for this model
        all_tokens (list): List of all token counts across models for normalization
    
    Returns:
        float: Efficiency score (0-1), higher is better
    """
    if not all_tokens or len(all_tokens) < 2:
        # If we only have one model or no token data, return accuracy
        return accuracy
    
    min_tokens = min(all_tokens)
    max_tokens = max(all_tokens)
    
    # Avoid division by zero
    if max_tokens == min_tokens:
        token_efficiency = 1.0
    else:
        normalized_tokens = (tokens - min_tokens) / (max_tokens - min_tokens)
        token_efficiency = 1 - normalized_tokens
    
    # Ensure token_efficiency is in [0, 1]
    token_efficiency = max(0, min(1, token_efficiency))
    
    # Handle edge cases
    if accuracy == 0 and token_efficiency == 0:
        return 0
    
    if accuracy + token_efficiency == 0:
        return 0
    
    # Harmonic mean of accuracy and token efficiency
    efficiency_score = 2 * (accuracy * token_efficiency) / (accuracy + token_efficiency)
    
    return round(efficiency_score, 4)

def calculate_overthinking_ratio(accuracy, tokens, baseline_tokens=50):
    """
    Calculate overthinking ratio: how many extra tokens per accuracy point.
    
    Overthinking Ratio = (tokens - baseline) / accuracy
    
    Lower values indicate less overthinking (more efficient).
    
    Args:
        accuracy (float): Model accuracy (0-1)
        tokens (float): Average tokens for this model
        baseline_tokens (int): Baseline token count for simple responses
    
    Returns:
        float: Overthinking ratio, lower is better
    """
    if accuracy == 0:
        return float('inf')  # Infinite overthinking if no accuracy
    
    extra_tokens = max(0, tokens - baseline_tokens)
    return round(extra_tokens / accuracy, 2)

def format_report_table(report_data):
    """Create pretty-printed table from report data with efficiency metrics"""
    headers = [
        "Task", "Accuracy", "Efficiency Score", "Overthinking Ratio",
        "Instruction Followed", "Tokens", "Chars", "Words", 
    ]
    
    rows = []
    
    # Collect all token means for normalization
    all_token_means = [metrics['output_tokens']['mean'] for metrics in report_data.values()]
    
    for size, metrics in report_data.items():
        # Calculate efficiency score
        efficiency_score = calculate_efficiency_score(
            metrics['accuracy']['mean'],
            metrics['output_tokens']['mean'],
            all_token_means
        )
        
        # Calculate overthinking ratio
        overthinking_ratio = calculate_overthinking_ratio(
            metrics['accuracy']['mean'],
            metrics['output_tokens']['mean']
        )
        
        rows.append([
            size,
            f"{metrics['accuracy']['mean']*100:.2f}% ± " + f"{metrics['accuracy']['std']*100:.2f}",
            f"{efficiency_score:.3f}",
            f"{overthinking_ratio:.1f}" if overthinking_ratio != float('inf') else "∞",
            f"{metrics['instruction_followed']['mean']*100:.2f}% ± " + f"{metrics['instruction_followed']['std']*100:.2f}",
            f"{metrics['output_tokens']['mean']:.1f} ± " + (f"{metrics['output_tokens']['std']:.1f}" if metrics['output_tokens']['std'] else "0"),
            f"{metrics['response_length']['mean']:.1f} ± " + (f"{metrics['response_length']['std']:.1f}" if metrics['response_length']['std'] else "0"),
            f"{metrics['word_count']['mean']:.1f} ± " + (f"{metrics['word_count']['std']:.1f}" if metrics['word_count']['std'] else "0")
        ])
    
    return tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f")

def generate_efficiency_ranking(report_data):
    """Generate ranking based on efficiency score"""
    rankings = []
    
    # Collect all token means for normalization
    all_token_means = [metrics['output_tokens']['mean'] for metrics in report_data.values()]
    
    for task_name, metrics in report_data.items():
        efficiency_score = calculate_efficiency_score(
            metrics['accuracy']['mean'],
            metrics['output_tokens']['mean'],
            all_token_means
        )
        
        overthinking_ratio = calculate_overthinking_ratio(
            metrics['accuracy']['mean'],
            metrics['output_tokens']['mean']
        )
        
        rankings.append({
            'task': task_name,
            'accuracy': metrics['accuracy']['mean'],
            'tokens': metrics['output_tokens']['mean'],
            'efficiency_score': efficiency_score,
            'overthinking_ratio': overthinking_ratio
        })
    
    # Sort by efficiency score (descending - higher is better)
    rankings.sort(key=lambda x: x['efficiency_score'], reverse=True)
    
    return rankings

def format_efficiency_ranking_table(rankings):
    """Format efficiency ranking table"""
    headers = ["Rank", "Task", "Efficiency Score", "Accuracy", "Tokens", "Overthinking Ratio"]
    
    rows = []
    for i, item in enumerate(rankings, 1):
        overthinking_str = f"{item['overthinking_ratio']:.1f}" if item['overthinking_ratio'] != float('inf') else "∞"
        rows.append([
            i,
            item['task'],
            f"{item['efficiency_score']:.3f}",
            f"{item['accuracy']*100:.1f}%",
            f"{item['tokens']:.1f}",
            overthinking_str
        ])
    
    return tabulate(rows, headers=headers, tablefmt="grid")

def generate_final_report(all_metrics, list_sizes, output_dir):
    """Generate aggregated report across all test cases with efficiency metrics"""
    report = {}
    tasks = {}
    for metric in all_metrics:
        task_name = metric.get('task', 'unknown')
        if task_name not in tasks:
            tasks[task_name] = []
        tasks[task_name].append(metric)
    
    # Process metrics for each task
    for task_name, task_metrics in tasks.items():
        # Process list-based tasks by list size
        # Special handling for tasks that have list sizes
        list_based_tasks = ['sorting', 'find_maximum', 'find_minimum', 'sum', 'multiplication', 
                           'odd_count', 'even_count', 'mean', 'median', 'mode']
        
        if task_name in list_based_tasks:
            # Group by the actual list size in the metrics
            size_groups = {}
            for metric in task_metrics:
                size = metric.get('list_size')
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(metric)

            # Process each size group
            for size, size_metrics in size_groups.items():
                if not size_metrics:
                    continue
                    
                task_size_name = f"{task_name}_{size}"
                
                report[task_size_name] = {
                    'accuracy': {
                        'mean': round(np.mean([m['accuracy'] for m in size_metrics]), 4),
                        'std': round(np.std([m['accuracy'] for m in size_metrics]), 4) if len(size_metrics) > 1 else 0
                    },
                    'instruction_followed': {
                        'mean': round(np.mean([m['instruction_followed_pct'] for m in size_metrics]), 4),
                        'std': round(np.std([m['instruction_followed_pct'] for m in size_metrics]), 4) if len(size_metrics) > 1 else 0
                    },
                    'response_length': {
                        "mean": round(np.mean([m['response_lengths'] for m in size_metrics]), 2),
                        "std": round(np.std([m['response_lengths'] for m in size_metrics]), 4) if len(size_metrics) > 1 else 0
                    },
                    "word_count":{
                        "mean": round(np.mean([m['word_counts'] for m in size_metrics]), 2),
                        "std": round(np.std([m['word_counts'] for m in size_metrics]), 4) if len(size_metrics) > 1 else 0
                    },
                    "output_tokens":{
                        "mean": round(np.mean([m['output_tokens'] for m in size_metrics]), 2),
                        "std": round(np.std([m['output_tokens'] for m in size_metrics]), 4) if len(size_metrics) > 1 else 0
                    }
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
                'response_length': {
                    "mean": round(np.mean([m['avg_response_length'] for m in task_metrics]), 2),
                    "std": round(np.std([m['avg_response_length'] for m in task_metrics]), 4) if len(task_metrics) > 1 else 0
                },
                "word_count":{
                    "mean": round(np.mean([m['avg_word_count'] for m in task_metrics]), 2),
                    "std": round(np.std([m['avg_word_count'] for m in task_metrics]), 4) if len(task_metrics) > 1 else 0
                },
                "output_tokens":{
                    "mean": round(np.mean([m['avg_output_tokens'] for m in task_metrics]), 2),
                    "std": round(np.std([m['avg_output_tokens'] for m in task_metrics]), 4) if len(task_metrics) > 1 else 0
                }
            }
    
    # Generate efficiency rankings
    efficiency_rankings = generate_efficiency_ranking(report)
    
    # Save final report with efficiency metrics
    enhanced_report = {
        'task_metrics': report,
        'efficiency_rankings': efficiency_rankings,
        'methodology': {
            'efficiency_score': 'Harmonic mean of accuracy and token efficiency (1 - normalized_tokens)',
            'overthinking_ratio': 'Extra tokens per accuracy point, lower is better',
            'token_normalization': 'Min-max normalization across all tasks'
        }
    }
    
    report_file = os.path.join(output_dir, "final_report.json")
    with open(report_file, 'w') as f:
        json.dump(enhanced_report, f, indent=2)
    
    # Create and save formatted table
    table = format_report_table(report)
    table_file = os.path.join(output_dir, "results_table.txt")
    with open(table_file, 'w') as f:
        f.write("📊 LLMThinkBench Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(table)
        f.write("\n\n")
        
        # Add efficiency ranking
        f.write("🏆 Efficiency Rankings (Best Balance of Accuracy & Conciseness)\n")
        f.write("=" * 70 + "\n\n")
        ranking_table = format_efficiency_ranking_table(efficiency_rankings)
        f.write(ranking_table)
        f.write("\n\n")
        
        # Add methodology explanation
        f.write("📋 Methodology\n")
        f.write("=" * 20 + "\n")
        f.write("• Efficiency Score: Harmonic mean of accuracy and token efficiency\n")
        f.write("• Token Efficiency: 1 - (normalized_tokens), where normalization uses min-max scaling\n")
        f.write("• Overthinking Ratio: (tokens - baseline) / accuracy, lower values indicate less overthinking\n")
        f.write("• Rankings prioritize models that achieve high accuracy with minimal tokens\n")
    
    logging.info(f"\n📊 Task Performance Summary:\n{table}")
    logging.info(f"\n🏆 Efficiency Rankings:\n{format_efficiency_ranking_table(efficiency_rankings)}")
    logging.info(f"\nSaved enhanced report to {report_file}")
    
    return enhanced_report