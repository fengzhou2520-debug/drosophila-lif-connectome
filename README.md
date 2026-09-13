# Drosophila Male-CNS Leaky Integrate-and-Fire Simulation

A complete, biophysically-grounded simulation of the *Drosophila melanogaster* male central nervous system (CNS) using the full connectome from Janelia's FlyEM project, with leaky integrate-and-fire neuronal models and full synaptic connectivity.

## Overview

This project implements a high-fidelity neural circuit simulation based on:

- **Full connectome**: Janelia male-CNS connectome (whole-brain connectivity at synaptic resolution)
- **Neuronal model**: Leaky Integrate-and-Fire (LIF) neurons with realistic biophysical parameters
- **Synaptic dynamics**: Neurotransmitter-specific synapses (glutamate, GABA, acetylcholine, etc.)
- **Neuromodulation**: Support for different neurotransmitter types and receptor kinetics
- **Network structure**: ~100,000 neurons with ~10 million synaptic connections

## Data Source

Connectome data is downloaded from [Janelia's male-CNS dataset](https://www.janelia.org/project-team/flyem/male-cns-connectome):

- **Neuron annotations**: `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- **Neurotransmitter data**: `body-neurotransmitters-male-cns-v1.0.feather`
- **Synaptic weights**: `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

## Project Structure

```
├── data/                              # Connectome data (auto-downloaded)
│   ├── raw/                          # Raw feather files
│   └── processed/                    # Preprocessed network data
├── src/
│   ├── data_loader.py               # Janelia API and file I/O
│   ├── connectome.py                # Connectome graph construction
│   ├── neuron.py                    # LIF neuron model implementation
│   ├── synapse.py                   # Synaptic dynamics and transmitters
│   ├── network.py                   # Network assembly and management
│   ├── simulator.py                 # Main simulation engine
│   ├── analysis.py                  # Network analysis and statistics
│   └── visualization.py             # Plotting and visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Load and explore connectome
│   ├── 02_network_construction.ipynb # Build network from connectome
│   ├── 03_run_simulation.ipynb       # Execute LIF simulation
│   └── 04_analysis.ipynb            # Analyze results
├── config/
│   ├── lif_params.yaml              # LIF neuron parameters
│   ├── synapse_params.yaml          # Synaptic parameters by transmitter
│   └── simulation.yaml              # Simulation configuration
├── tests/
│   └── test_*.py                    # Unit tests
└── examples/
    ├── basic_simulation.py          # Minimal working example
    ├── parameter_sweep.py           # Systematic parameter exploration
    └── stimulus_response.py          # Input-output analysis
```

## Installation

```bash
git clone https://github.com/fengzhou2520-debug/drosophila-lif-connectome.git
cd drosophila-lif-connectome
pip install -r requirements.txt
```

## Quick Start

### 1. Download Connectome Data

```python
from src.data_loader import ConnectomeDataLoader

loader = ConnectomeDataLoader()
loader.download_janelia_data()
loader.load_feather_files()
```

### 2. Build Network

```python
from src.connectome import Connectome
from src.network import FlyBrainNetwork

connectome = Connectome(loader.neurons, loader.synapses, loader.neurotransmitters)
network = FlyBrainNetwork(connectome)
network.build()
```

### 3. Run Simulation

```python
from src.simulator import LIFSimulator
from src.stimulus import PoissonStimulus

stimulus = PoissonStimulus(rate=5.0)  # 5 Hz Poisson input
simulator = LIFSimulator(network, dt=0.1)  # 0.1 ms timestep
spikes = simulator.run(duration=1000.0, stimulus=stimulus)
```

### 4. Analyze Results

```python
from src.analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer(network, spikes)
analyzer.compute_firing_rates()
analyzer.compute_correlations()
analyzer.plot_raster()
```

## References

1. **Connectome Publication**: Janelia FlyEM male-CNS connectome  
   https://www.janelia.org/project-team/flyem/male-cns-connectome

2. **Related Work**:
   - Shiu et al., Drosophila brain model: [GitHub](https://github.com/philshiu/Drosophila_brain_model)
   - Eonsystems fly-brain: [GitHub](https://github.com/eonsystemspbc/fly-brain)
   - Nature 2024: https://doi.org/10.1038/s41586-024-07763-9
   - PMC paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10187186/

3. **Theoretical Background**:
   - Hodgkin-Huxley model
   - Integrate-and-fire neuron models
   - Synaptic plasticity and short-term dynamics

## Features

- ✅ Full connectome parsing and validation
- ✅ Biophysical LIF neuron model with leak conductance
- ✅ Neurotransmitter-specific synaptic models (iGluR, nAChR, GABA-A/B)
- ✅ Short-term synaptic plasticity (depression and facilitation)
- ✅ Efficient sparse matrix representation
- ✅ Multi-threaded spike detection and propagation
- ✅ Detailed network statistics and visualization
- ✅ Input stimulus generation (Poisson, step, sinusoid)
- ✅ Spike-time raster plots and autocorrelograms
- ✅ Population activity analysis

## Configuration

Edit `config/` files to customize:
- Membrane time constants and resting potentials
- Synaptic weights and transmission delays
- Spike threshold and reset potential
- Refractory period parameters
- Neurotransmitter dynamics

## Performance

Expected runtime (approximate):
- **Data download**: ~10-15 minutes
- **Network construction**: ~5-10 minutes
- **1-second simulation** (100 Hz resolution): ~30-60 minutes (CPU), ~5-10 minutes (GPU if enabled)

## Contributing

This project welcomes contributions. Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Janelia Research Campus for the high-quality connectome data
- The Drosophila neuroscience community for reference implementations
- Contributors to Brian2, NumPy, and scientific Python ecosystem
