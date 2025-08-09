#!/usr/bin/env python3
"""
Overnight particle swarm optimization runner.
This script is designed for SSH runs with comprehensive logging and error handling.
"""

import os
import sys
import signal
import traceback
from datetime import datetime  # This is correct
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for SSH

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main functionality from pitch_perts
from pitch_perts import (
    params_obj, target_response, pert_signal, T_sim, 
    truncate_start, truncate_end, sensor_processor, 
    system_choice, calibrate_opt, pitch_pert_data, fig_save_path,
    run_calibration, run_simulation
)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def run_overnight_optimization():
    """Run overnight optimization with enhanced settings and logging."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting overnight particle swarm optimization")
    print(f"Output directory: {fig_save_path}")
    
    try:
        # Override calibration settings for overnight runs
        original_calibrate_opt = calibrate_opt
        original_particle_size = params_obj.cal_set_dict.get('particle_size', 100)
        original_iterations = params_obj.cal_set_dict.get('iterations', 1000)
        original_runs = params_obj.cal_set_dict.get('runs', 10)
        
        # # Set conservative settings for overnight runs
        # params_obj.cal_set_dict['particle_size'] = 10
        # params_obj.cal_set_dict['iterations'] = 50
        # params_obj.cal_set_dict['runs'] = 10
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using overnight settings:")
        print(f"  - Particles: {params_obj.cal_set_dict['particle_size']} (original: {original_particle_size})")
        print(f"  - Iterations: {params_obj.cal_set_dict['iterations']} (original: {original_iterations})")
        print(f"  - Runs: {params_obj.cal_set_dict['runs']} (original: {original_runs})")
        
        # Run calibration using the shared function
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting particle swarm calibration...")
        
        cal_params, mse_history, run_dir, sensor_delay_aud, sensor_delay_som, actuator_delay, pitch_pert_data = run_calibration(
            original_calibrate_opt,
            params_obj, 
            target_response, 
            pert_signal, 
            T_sim, 
            truncate_start, 
            truncate_end, 
            sensor_processor
        )

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Optimization completed successfully!")
        print(f"Final RMSE: {mse_history}")
        print(f"Results saved to: {run_dir}")
        
        # Run simulation using the shared function
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running final simulation with optimized parameters...")
        
        system, timeseries_truncated, system_response_truncated, aud_pert_truncated = run_simulation(
            cal_params, 
            pert_signal, 
            T_sim, 
            truncate_start, 
            truncate_end, 
            sensor_processor, 
            system_choice, 
            sensor_delay_aud, 
            sensor_delay_som, 
            actuator_delay, 
            run_dir, 
            pitch_pert_data
        )
        
        # Create final plots
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating final plots...")
        system.plot_data_overlay(
            'abs2sens', 
            target_response, 
            pitch_pert_data, 
            time_trunc=timeseries_truncated, 
            resp_trunc=system_response_truncated, 
            pitch_pert_truncated=aud_pert_truncated, 
            output_dir=run_dir
        )
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final simulation and plots completed!")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Overnight optimization run completed successfully!")
        
        # Restore original settings
        params_obj.cal_set_dict['particle_size'] = original_particle_size
        params_obj.cal_set_dict['iterations'] = original_iterations
        params_obj.cal_set_dict['runs'] = original_runs
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Optimization failed!")
        print(f"Error details: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
        # Save error information to the timestamped folder if available, otherwise to main output directory
        error_dir = run_dir if 'run_dir' in locals() else fig_save_path
        error_file = os.path.join(error_dir, f"optimization_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_file, 'w') as f:
            f.write(f"Optimization failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        
        print(f"Error details saved to: {error_file}")
        sys.exit(1)

def main():
    """Main function to run overnight optimization."""
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    run_overnight_optimization()

if __name__ == "__main__":
    main() 