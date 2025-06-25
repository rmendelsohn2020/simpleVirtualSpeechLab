#!/usr/bin/env python3
"""
Monitoring script for particle swarm optimization runs.
Use this to check progress during overnight SSH runs.
"""

import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def find_latest_run(output_dir):
    """Find the most recent optimization run directory."""
    if not os.path.exists(output_dir):
        return None
    
    run_dirs = [d for d in os.listdir(output_dir) if d.startswith('particle_swarm_run_')]
    if not run_dirs:
        return None
    
    # Sort by timestamp and return the latest
    run_dirs.sort()
    return os.path.join(output_dir, run_dirs[-1])

def read_progress_data(run_dir):
    """Read progress data from the run directory."""
    progress_file = os.path.join(run_dir, "progress_history.json")
    if not os.path.exists(progress_file):
        return None
    
    with open(progress_file, 'r') as f:
        return json.load(f)

def read_log_file(run_dir, num_lines=20):
    """Read the last N lines from the log file."""
    log_file = os.path.join(run_dir, "optimization_log.txt")
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        return lines[-num_lines:] if len(lines) > num_lines else lines

def estimate_remaining_time(progress_data):
    """Estimate remaining time based on completed runs."""
    if not progress_data or not progress_data['run_summaries']:
        return None
    
    completed_runs = len(progress_data['run_summaries'])
    total_runs = max(progress_data['runs']) if progress_data['runs'] else 0
    
    if completed_runs == 0 or total_runs == 0:
        return None
    
    # Calculate average time per run
    durations = [summary['duration_seconds'] for summary in progress_data['run_summaries']]
    avg_duration = np.mean(durations)
    
    remaining_runs = total_runs - completed_runs
    estimated_remaining = remaining_runs * avg_duration
    
    return {
        'completed_runs': completed_runs,
        'total_runs': total_runs,
        'avg_duration_per_run': avg_duration,
        'estimated_remaining_seconds': estimated_remaining,
        'estimated_remaining_hours': estimated_remaining / 3600
    }

def print_status_summary(run_dir, progress_data):
    """Print a summary of the current optimization status."""
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION STATUS SUMMARY")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not progress_data:
        print("No progress data found.")
        return
    
    # Overall progress
    if progress_data['overall_best_rmse']:
        best_rmse = min(progress_data['overall_best_rmse'])
        print(f"Best RMSE so far: {best_rmse:.6f}")
    
    # Run progress
    completed_runs = len(progress_data['run_summaries'])
    total_runs = max(progress_data['runs']) if progress_data['runs'] else 0
    print(f"Progress: {completed_runs}/{total_runs} runs completed")
    
    # Time estimation
    time_est = estimate_remaining_time(progress_data)
    if time_est:
        print(f"Average time per run: {time_est['avg_duration_per_run']:.1f}s")
        print(f"Estimated remaining time: {time_est['estimated_remaining_hours']:.1f} hours")
    
    # Recent runs summary
    if progress_data['run_summaries']:
        print(f"\nRecent runs:")
        for summary in progress_data['run_summaries'][-5:]:  # Last 5 runs
            status = "✓" if summary['convergence_info']['converged'] else "✗"
            print(f"  Run {summary['run_number']}: RMSE={summary['final_rmse']:.6f}, "
                  f"Duration={summary['duration_seconds']:.1f}s {status}")

def print_recent_logs(run_dir, num_lines=10):
    """Print recent log entries."""
    log_lines = read_log_file(run_dir, num_lines)
    if log_lines:
        print(f"\n{'='*60}")
        print(f"RECENT LOG ENTRIES (last {len(log_lines)} lines)")
        print(f"{'='*60}")
        for line in log_lines:
            print(line.rstrip())
    else:
        print("No log entries found.")

def create_live_plot(run_dir, progress_data, save_plot=True):
    """Create a live plot of the optimization progress."""
    if not progress_data or not progress_data['overall_best_rmse']:
        print("No progress data available for plotting.")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Overall best RMSE progress
    plt.subplot(2, 3, 1)
    plt.plot(progress_data['runs'], progress_data['overall_best_rmse'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Run Number')
    plt.ylabel('Best RMSE')
    plt.title('Overall Best RMSE Progress')
    plt.grid(True)
    
    # Plot 2: Individual run final RMSEs
    plt.subplot(2, 3, 2)
    if progress_data['run_summaries']:
        run_numbers = [summary['run_number'] for summary in progress_data['run_summaries']]
        final_rmses = [summary['final_rmse'] for summary in progress_data['run_summaries']]
        colors = ['green' if summary['convergence_info']['converged'] else 'red' 
                 for summary in progress_data['run_summaries']]
        plt.bar(run_numbers, final_rmses, color=colors, alpha=0.7)
        plt.xlabel('Run Number')
        plt.ylabel('Final RMSE')
        plt.title('Final RMSE by Run (Green=Converged)')
        plt.grid(True)
    
    # Plot 3: Run durations
    plt.subplot(2, 3, 3)
    if progress_data['run_summaries']:
        durations = [summary['duration_seconds'] for summary in progress_data['run_summaries']]
        plt.bar(run_numbers, durations, alpha=0.7, color='blue')
        plt.xlabel('Run Number')
        plt.ylabel('Duration (seconds)')
        plt.title('Run Duration')
        plt.grid(True)
    
    # Plot 4: Convergence status
    plt.subplot(2, 3, 4)
    if progress_data['run_summaries']:
        converged = [1 if summary['convergence_info']['converged'] else 0 
                    for summary in progress_data['run_summaries']]
        plt.bar(run_numbers, converged, alpha=0.7, color='orange')
        plt.xlabel('Run Number')
        plt.ylabel('Converged (1=Yes, 0=No)')
        plt.title('Convergence Status')
        plt.grid(True)
        plt.ylim(0, 1.2)
    
    # Plot 5: RMSE improvement over time
    plt.subplot(2, 3, 5)
    if len(progress_data['overall_best_rmse']) > 1:
        improvements = []
        for i in range(1, len(progress_data['overall_best_rmse'])):
            improvement = progress_data['overall_best_rmse'][i-1] - progress_data['overall_best_rmse'][i]
            improvements.append(improvement)
        
        plt.bar(range(1, len(improvements)+1), improvements, alpha=0.7, color='purple')
        plt.xlabel('Run Number')
        plt.ylabel('RMSE Improvement')
        plt.title('RMSE Improvement per Run')
        plt.grid(True)
    
    # Plot 6: Cumulative time
    plt.subplot(2, 3, 6)
    if progress_data['run_summaries']:
        cumulative_time = np.cumsum([summary['duration_seconds'] for summary in progress_data['run_summaries']])
        plt.plot(run_numbers, cumulative_time / 3600, 'g-', linewidth=2)  # Convert to hours
        plt.xlabel('Run Number')
        plt.ylabel('Cumulative Time (hours)')
        plt.title('Cumulative Runtime')
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_plot:
        plot_file = os.path.join(run_dir, "live_progress_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Live plot saved to: {plot_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Monitor particle swarm optimization progress')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (default: uses config)')
    parser.add_argument('--run-dir', type=str, default=None,
                       help='Specific run directory to monitor')
    parser.add_argument('--log-lines', type=int, default=10,
                       help='Number of recent log lines to show')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip creating live plot')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode: continuously monitor progress')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds for watch mode')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        from utils.get_configs import get_paths
        path_obj = get_paths()
        output_dir = path_obj.fig_save_path
    else:
        output_dir = args.output_dir
    
    # Determine run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(output_dir)
    
    if not run_dir or not os.path.exists(run_dir):
        print(f"Error: No optimization run found in {output_dir}")
        return
    
    if args.watch:
        print(f"Watching optimization progress in {run_dir}")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                progress_data = read_progress_data(run_dir)
                print_status_summary(run_dir, progress_data)
                print_recent_logs(run_dir, args.log_lines)
                
                if not args.no_plot:
                    create_live_plot(run_dir, progress_data, save_plot=True)
                
                import time
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        # Single status check
        progress_data = read_progress_data(run_dir)
        print_status_summary(run_dir, progress_data)
        print_recent_logs(run_dir, args.log_lines)
        
        if not args.no_plot:
            create_live_plot(run_dir, progress_data, save_plot=True)

if __name__ == "__main__":
    main() 