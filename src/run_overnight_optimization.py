#!/usr/bin/env python3
"""
Overnight particle swarm optimization runner.
This script is designed for SSH runs with comprehensive logging and error handling.
"""

import os
import sys
import signal
import traceback
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pitch_perts import *
from utils.get_configs import get_paths

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def main():
    """Main function to run overnight optimization."""
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get paths
    path_obj = get_paths()
    output_dir = path_obj.fig_save_path
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting overnight particle swarm optimization")
    print(f"Output directory: {output_dir}")
    
    try:
        # Set calibration option to Particle Swarm
        calibrate_opt = 'Particle Swarm'
        
        # Create calibrator with enhanced settings for overnight runs
        calibrator = PitchPertCalibrator(
            params_obj=params_obj,
            target_response=target_response,
            pert_signal=pert_signal,
            T_sim=T_sim,
            truncate_start=truncate_start,
            truncate_end=truncate_end,
            sensor_processor=RelativeSensorProcessor()
        )

        # Run optimization with conservative settings for overnight runs
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting particle swarm calibration...")
        
        cal_params, mse_history, run_dir = calibrator.particle_swarm_calibrate(
            num_particles=10,
            max_iters=50,
            convergence_tol=0.01,
            runs=10,
            log_interval=20,  # Log every 20 iterations
            save_interval=100,  # Save intermediate results every 100 iterations
            output_dir=output_dir
        )

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Optimization completed successfully!")
        print(f"Final RMSE: {mse_history}")
        print(f"Results saved to: {run_dir}")
        
        # Extract delays
        sensor_delay_aud = int(cal_params.sensor_delay_aud)
        sensor_delay_som = int(cal_params.sensor_delay_som)
        actuator_delay = int(cal_params.actuator_delay)

        # Save final results to the timestamped folder
        readout_optimized_params(cal_params, sensor_delay_aud, sensor_delay_som, actuator_delay, output_dir=run_dir)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final results saved successfully!")
        
        # Run final simulation with optimized parameters
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running final simulation with optimized parameters...")
        
        system = Controller(
            sensor_processor=RelativeSensorProcessor(), 
            input_A=cal_params.A_init, 
            input_B=cal_params.B_init, 
            input_C=cal_params.C_aud_init, 
            ref_type=params_obj.ref_type, 
            dist_custom=pert_signal.signal, 
            dist_type=['Auditory'], 
            K_vals=[cal_params.K_aud_init, cal_params.K_som_init], 
            L_vals=[cal_params.L_aud_init, cal_params.L_som_init], 
            Kf_vals=[cal_params.Kf_aud_init, cal_params.Kf_som_init], 
            timeseries=T_sim
        )
        
        system.simulate_with_2sensors(
            delta_t_s_aud=sensor_delay_aud, 
            delta_t_s_som=sensor_delay_som, 
            delta_t_a=actuator_delay
        )

        # Create final plots and save to the timestamped folder
        timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)
        aud_pert_truncated = truncate_data(T_sim, system.v_aud, truncate_start, truncate_end)[1]
        
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
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Optimization failed!")
        print(f"Error details: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
        # Save error information to the timestamped folder if available, otherwise to main output directory
        error_dir = run_dir if 'run_dir' in locals() else output_dir
        error_file = os.path.join(error_dir, f"optimization_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_file, 'w') as f:
            f.write(f"Optimization failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        
        print(f"Error details saved to: {error_file}")
        sys.exit(1)

if __name__ == "__main__":
    main() 