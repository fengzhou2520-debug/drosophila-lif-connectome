"""
Basic example: Load connectome, build network, run LIF simulation, and analyze results.

This is a minimal working example demonstrating the full pipeline:
1. Load Janelia male-CNS connectome data
2. Construct network with LIF neurons
3. Run simulation with Poisson input
4. Analyze and visualize results
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import ConnectomeDataLoader
from connectome import Connectome, FlyBrainNetwork
from neuron import LIFParameters
from simulator import LIFSimulator, PoissonStimulus
from analysis import NetworkAnalyzer
from visualization import SimulationPlotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run basic simulation example."""
    
    logger.info("=" * 70)
    logger.info("Drosophila Male-CNS LIF Simulation - Basic Example")
    logger.info("=" * 70)
    
    # Step 1: Load connectome data
    logger.info("\n[1/5] Loading connectome data from Janelia...")
    loader = ConnectomeDataLoader(data_dir="data")
    
    try:
        loader.load_feather_files()
        loader.summary()
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.error("Please download from: https://male-cns.janelia.org/download/")
        logger.error("Required files:")
        logger.error("  - body-annotations-male-cns-v1.0-minconf-0.5.feather")
        logger.error("  - body-neurotransmitters-male-cns-v1.0.feather")
        logger.error("  - connectome-weights-male-cns-v1.0-minconf-0.5.feather")
        return
    
    # Step 2: Build connectome
    logger.info("\n[2/5] Constructing connectome graph...")
    connectome = Connectome(
        neurons_df=loader.neurons,
        synapses_df=loader.synapses,
        neurotransmitters_df=loader.neurotransmitters,
    )
    
    # Step 3: Create network with LIF neurons
    logger.info("\n[3/5] Building LIF neural network...")
    params = LIFParameters.drosophila_defaults()
    network = FlyBrainNetwork(connectome, neuron_params=params, dt=0.1)
    network.build(verbose=True)
    
    # Step 4: Run simulation
    logger.info("\n[4/5] Running LIF simulation...")
    
    # Create stimulus: Poisson input to 10% of neurons at 5 Hz
    stimulus = PoissonStimulus(rate=5.0, amplitude=50.0, fraction=0.1)
    
    # Run 1 second simulation
    simulator = LIFSimulator(
        network,
        dt=0.1,
        record_spikes=True,
        record_voltage=False,  # Set to True for detailed voltage traces
        record_current=False,
    )
    
    results = simulator.run(
        duration=1000.0,  # 1 second
        stimulus=stimulus,
        verbose=True,
    )
    
    # Step 5: Analyze results
    logger.info("\n[5/5] Analyzing simulation results...")
    analyzer = NetworkAnalyzer(network, results)
    logger.info(analyzer.summary())
    
    # Print additional statistics
    fire_rates = analyzer.compute_firing_rates()
    rate_dist = analyzer.compute_firing_rate_distribution()
    logger.info(f"\nFiring rate statistics (Hz):")
    logger.info(f"  Mean: {rate_dist.get('mean', 0):.2f}")
    logger.info(f"  Median: {rate_dist.get('median', 0):.2f}")
    logger.info(f"  Max: {rate_dist.get('max', 0):.2f}")
    
    # Connectivity analysis
    conn_analyzer = analyzer  # Reuse for connectivity if needed
    logger.info(f"\nNetwork connectivity:")
    logger.info(f"  Total synapses: {len(network.synapse_list):,}")
    logger.info(f"  Network density: {len(network.synapse_list) / (len(network.neurons)**2):.2e}")
    
    # Step 6: Visualization
    logger.info("\nGenerating visualizations...")
    plotter = SimulationPlotter(figsize=(16, 12))
    
    # Create summary figure
    fig, axes = plotter.plot_network_summary(
        spike_times=results["spike_times"],
        duration=results["duration"],
        window_size=10.0,
    )
    
    output_file = "simulation_results.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {output_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Simulation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
