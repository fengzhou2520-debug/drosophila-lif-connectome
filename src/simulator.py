"""
Main simulation engine for LIF network dynamics.

Orchestrates:
- Time stepping and neural integration
- Spike detection and propagation
- Synaptic current computation
- Recording and data collection
- Stimulus delivery
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
from collections import defaultdict

from .connectome import FlyBrainNetwork
from .neuron import LIFNeuronPopulation
from .synapse import SynapsePopulation

logger = logging.getLogger(__name__)


class Stimulus:
    """Base class for input stimuli."""
    
    def get_current(self, time: float, neuron_ids: np.ndarray) -> np.ndarray:
        """
        Get stimulus current for neurons at given time.
        
        Args:
            time: Current time (ms)
            neuron_ids: Array of neuron IDs
            
        Returns:
            Current array (pA) for each neuron
        """
        raise NotImplementedError


class NoStimulus(Stimulus):
    """No external stimulus."""
    
    def get_current(self, time: float, neuron_ids: np.ndarray) -> np.ndarray:
        return np.zeros(len(neuron_ids), dtype=np.float32)


class PoissonStimulus(Stimulus):
    """Poisson spike train input to random neurons."""
    
    def __init__(self, rate: float = 5.0, amplitude: float = 100.0, fraction: float = 0.1):
        """
        Initialize Poisson stimulus.
        
        Args:
            rate: Spike rate (Hz)
            amplitude: Current amplitude per spike (pA)
            fraction: Fraction of neurons receiving input
        """
        self.rate = rate
        self.amplitude = amplitude
        self.fraction = fraction
        self.last_spike_times = {}
    
    def get_current(self, time: float, neuron_ids: np.ndarray) -> np.ndarray:
        """Poisson-distributed spike train input."""
        currents = np.zeros(len(neuron_ids), dtype=np.float32)
        
        # Randomly select neurons to stimulate
        n_stimulated = max(1, int(len(neuron_ids) * self.fraction))
        stim_indices = np.random.choice(len(neuron_ids), size=n_stimulated, replace=False)
        
        # Generate Poisson spikes
        dt = 0.1  # Assumed timestep
        spike_prob = self.rate * dt / 1000.0  # Convert Hz to probability
        
        for idx in stim_indices:
            if np.random.random() < spike_prob:
                currents[idx] = self.amplitude
        
        return currents


class StepStimulus(Stimulus):
    """Step current input to selected neurons."""
    
    def __init__(self, amplitude: float = 100.0, start_time: float = 100.0, 
                 end_time: float = 900.0, neuron_indices: Optional[List[int]] = None):
        """
        Initialize step stimulus.
        
        Args:
            amplitude: Current amplitude (pA)
            start_time: Stimulus start (ms)
            end_time: Stimulus end (ms)
            neuron_indices: Indices of stimulated neurons (None = all)
        """
        self.amplitude = amplitude
        self.start_time = start_time
        self.end_time = end_time
        self.neuron_indices = neuron_indices
    
    def get_current(self, time: float, neuron_ids: np.ndarray) -> np.ndarray:
        """Step current input."""
        currents = np.zeros(len(neuron_ids), dtype=np.float32)
        
        if self.start_time <= time <= self.end_time:
            if self.neuron_indices is None:
                currents[:] = self.amplitude
            else:
                for idx in self.neuron_indices:
                    if 0 <= idx < len(neuron_ids):
                        currents[idx] = self.amplitude
        
        return currents


class LIFSimulator:
    """Main simulation engine."""
    
    def __init__(
        self,
        network: FlyBrainNetwork,
        dt: float = 0.1,
        record_spikes: bool = True,
        record_voltage: bool = False,
        record_current: bool = False,
    ):
        """
        Initialize simulator.
        
        Args:
            network: FlyBrainNetwork instance (must be built)
            dt: Integration timestep (ms)
            record_spikes: Record spike times
            record_voltage: Record membrane potentials (memory intensive)
            record_current: Record synaptic currents
        """
        if not network.built:
            raise ValueError("Network must be built before simulation")
        
        self.network = network
        self.dt = dt
        self.record_spikes = record_spikes
        self.record_voltage = record_voltage
        self.record_current = record_current
        
        # Simulation state
        self.time = 0.0
        self.spike_times = defaultdict(list)  # Maps neuron_id to spike times
        self.voltage_history = {}
        self.current_history = {}
        self.stimulus_history = {}
        
        logger.info(f"LIFSimulator initialized with dt={dt} ms")
    
    def run(
        self,
        duration: float,
        stimulus: Optional[Stimulus] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run simulation.
        
        Args:
            duration: Simulation duration (ms)
            stimulus: Input stimulus (None = no stimulus)
            verbose: Show progress bar
            
        Returns:
            Dictionary with results (spike_times, voltages, etc.)
        """
        if stimulus is None:
            stimulus = NoStimulus()
        
        logger.info(f"Starting simulation: {duration} ms")
        self.time = 0.0
        
        # Reset network
        self.network.neurons.reset()
        self.network.synapses.synapses = self.network.synapse_list
        
        # Initialize history
        if self.record_voltage:
            self.voltage_history = {i: [] for i in range(len(self.network.neurons))}
        if self.record_current:
            self.current_history = {i: [] for i in range(len(self.network.neurons))}
        
        # Integration loop
        n_steps = int(duration / self.dt)
        iterator = tqdm(range(n_steps), disable=not verbose)
        
        for step in iterator:
            self.time = step * self.dt
            
            # Get stimulus current
            I_stim = stimulus.get_current(self.time, self.network.connectome.neuron_ids)
            self.network.neurons.set_external_current(I_stim)
            
            # Compute synaptic currents
            I_syn = self.network.synapses.compute_postsynaptic_currents(
                self.network.connectome.neuron_ids,
                self.network.neurons.V,
                self.time,
            )
            self.network.neurons.set_synaptic_current(I_syn)
            
            # Neural integration step
            spiked = self.network.neurons.step(self.time)
            
            # Record spikes
            if self.record_spikes:
                spike_indices = np.where(spiked)[0]
                for idx in spike_indices:
                    neuron_id = self.network.connectome.neuron_ids[idx]
                    self.spike_times[neuron_id].append(self.time)
                    
                    # Deliver spikes to postsynaptic neurons
                    self.network.synapses.deliver_spike(neuron_id, self.time)
            
            # Record voltages
            if self.record_voltage and step % 10 == 0:  # Sample every 10 steps
                for i in range(len(self.network.neurons)):
                    self.voltage_history[i].append(self.network.neurons.V[i])
            
            # Record currents
            if self.record_current and step % 10 == 0:
                for i in range(len(self.network.neurons)):
                    self.current_history[i].append(self.network.neurons.I_syn[i])
            
            # Update progress
            if verbose and (step + 1) % max(1, n_steps // 20) == 0:
                total_spikes = sum(len(times) for times in self.spike_times.values())
                iterator.set_description(f"Spikes: {total_spikes}")
        
        logger.info(f"Simulation complete: {self.time:.1f} ms")
        
        return self._collect_results(duration)
    
    def _collect_results(self, duration: float) -> Dict:
        """Collect simulation results."""
        results = {
            "time": self.time,
            "duration": duration,
            "dt": self.dt,
            "spike_times": dict(self.spike_times),
        }
        
        if self.record_voltage:
            results["voltage_history"] = self.voltage_history
        
        if self.record_current:
            results["current_history"] = self.current_history
        
        # Compute statistics
        results["statistics"] = self._compute_statistics()
        
        return results
    
    def _compute_statistics(self) -> Dict:
        """Compute simulation statistics."""
        stats = {
            "total_spikes": sum(len(times) for times in self.spike_times.values()),
            "n_active_neurons": sum(1 for times in self.spike_times.values() if len(times) > 0),
        }
        
        # Firing rates
        firing_rates = []
        for times in self.spike_times.values():
            if times:
                duration_sec = self.time / 1000.0
                if duration_sec > 0:
                    firing_rates.append(len(times) / duration_sec)
        
        if firing_rates:
            stats["mean_firing_rate"] = np.mean(firing_rates)
            stats["max_firing_rate"] = np.max(firing_rates)
            stats["firing_rate_std"] = np.std(firing_rates)
        
        return stats
    
    def get_raster_data(self, neuron_sample: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spike raster data for plotting.
        
        Args:
            neuron_sample: Sample N neurons (None = all)
            
        Returns:
            (spike_times, spike_neuron_indices)
        """
        all_spike_times = []
        all_neuron_indices = []
        
        neuron_ids = list(self.spike_times.keys())
        if neuron_sample is not None and len(neuron_ids) > neuron_sample:
            neuron_ids = np.random.choice(neuron_ids, size=neuron_sample, replace=False)
        
        for idx, neuron_id in enumerate(neuron_ids):
            for spike_time in self.spike_times[neuron_id]:
                all_spike_times.append(spike_time)
                all_neuron_indices.append(idx)
        
        return np.array(all_spike_times), np.array(all_neuron_indices)
    
    def get_firing_rate_curve(self, window_size: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get population firing rate over time (smoothed).
        
        Args:
            window_size: Smoothing window (ms)
            
        Returns:
            (times, firing_rates)
        """
        # Create time bins
        n_bins = int(self.time / window_size)
        bin_edges = np.linspace(0, self.time, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        firing_rates = np.zeros(n_bins)
        
        # Count spikes in each bin
        for neuron_id, spike_times in self.spike_times.items():
            hist, _ = np.histogram(spike_times, bins=bin_edges)
            firing_rates += hist
        
        # Convert to Hz
        if len(self.network.neurons) > 0:
            firing_rates /= len(self.network.neurons) * (window_size / 1000.0)
        
        return bin_centers, firing_rates


if __name__ == "__main__":
    # Example usage (requires built network)
    logger.info("LIF Simulator module")
    logger.info("Use with: simulator = LIFSimulator(network)")
    logger.info("Then: results = simulator.run(duration=1000)")
