"""Example: Characterize network response to different input stimuli.

Compares network activity under different stimulus conditions:
- No stimulus (spontaneous activity)
- Step current injection
- Poisson spike train input
- Sinusoidal current
"""

import logging
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import ConnectomeDataLoader
from connectome import Connectome, FlyBrainNetwork
from neuron import LIFParameters
from simulator import LIFSimulator, StepStimulus, PoissonStimulus, NoStimulus
from analysis import NetworkAnalyzer
from visualization import SimulationPlotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_stimulus_comparison():
    """Run simulations with different stimuli and compare responses."""
    
    logger.info("Loading connectome and building network...")
    loader = ConnectomeDataLoader()
    
    try:
        loader.load_feather_files()
    except FileNotFoundError:
        logger.error("Data files not found. Download from https://male-cns.janelia.org/download/")
        return
    
    connectome = Connectome(loader.neurons, loader.synapses, loader.neurotransmitters)
    network = FlyBrainNetwork(connectome, neuron_params=LIFParameters.drosophila_defaults())
    network.build(verbose=False)
    
    # Define stimuli to test
    stimuli = {
        "none": NoStimulus(),
        "step_100pA": StepStimulus(amplitude=100.0, start_time=200, end_time=800),
        "poisson_5Hz": PoissonStimulus(rate=5.0, amplitude=50.0, fraction=0.1),
    }
    
    results = {}
    
    for stim_name, stimulus in stimuli.items():
        logger.info(f"\nRunning simulation with {stim_name}...")
        
        simulator = LIFSimulator(network, dt=0.1, record_spikes=True)
        sim_results = simulator.run(duration=1000.0, stimulus=stimulus, verbose=True)
        
        analyzer = NetworkAnalyzer(network, sim_results)
        fire_rates = analyzer.compute_firing_rates()
        rate_stats = analyzer.compute_firing_rate_distribution()
        
        results[stim_name] = {
            "spike_times": sim_results["spike_times"],
            "firing_rate_mean": rate_stats.get("mean", 0),
            "firing_rate_max": rate_stats.get("max", 0),
            "active_neurons": analyzer.get_active_neurons(),
        }
        
        logger.info(f"  Mean FR: {rate_stats.get('mean', 0):.2f} Hz")
        logger.info(f"  Max FR: {rate_stats.get('max', 0):.2f} Hz")
        logger.info(f"  Active neurons: {analyzer.get_active_neurons()}")
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("Stimulus Comparison")
    logger.info("="*60)
    
    for stim_name, data in results.items():
        logger.info(f"{stim_name}:")
        logger.info(f"  Mean firing rate: {data['firing_rate_mean']:.2f} Hz")
        logger.info(f"  Max firing rate: {data['firing_rate_max']:.2f} Hz")
        logger.info(f"  Active neurons: {data['active_neurons']}")


if __name__ == "__main__":
    run_stimulus_comparison()
