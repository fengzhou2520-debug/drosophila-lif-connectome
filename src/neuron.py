"""
Leaky Integrate-and-Fire (LIF) neuron model implementation.

The LIF model is described by the differential equation:
    C_m * dV/dt = g_leak * (E_leak - V) + I_syn + I_ext
    
where:
    - V: membrane potential (mV)
    - C_m: membrane capacitance (F/cm²)
    - g_leak: leak conductance (S/cm²)
    - E_leak: leak reversal potential (mV)
    - I_syn: synaptic current input (A/cm²)
    - I_ext: external current (A/cm²)

Spike generation:
    - When V > V_th (threshold), neuron fires a spike
    - V is reset to V_reset
    - Neuron enters refractory period where no spikes can occur
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LIFParameters:
    """Parameters for LIF neuron model."""
    
    # Membrane properties
    C_m: float = 1.0  # Membrane capacitance (μF/cm²)
    g_leak: float = 0.1  # Leak conductance (mS/cm²)
    E_leak: float = -70.0  # Leak reversal potential (mV)
    
    # Spike properties
    V_th: float = -55.0  # Spike threshold (mV)
    V_reset: float = -70.0  # Reset potential (mV)
    tau_m: float = 10.0  # Membrane time constant (ms)
    
    # Refractory period
    tau_ref: float = 2.0  # Absolute refractory period (ms)
    
    # Input noise (optional)
    sigma_noise: float = 0.0  # Standard deviation of noise current (pA)
    
    # Surface area (for scaling)
    area: float = 1000.0  # Surface area (μm²)
    
    @classmethod
    def drosophila_defaults(cls) -> "LIFParameters":
        """Default parameters based on Drosophila neuron physiology."""
        return cls(
            C_m=1.0,
            g_leak=0.1,
            E_leak=-65.0,
            V_th=-50.0,
            V_reset=-65.0,
            tau_m=15.0,
            tau_ref=1.5,
            sigma_noise=0.5,
            area=1000.0,
        )


class LIFNeuron:
    """Single leaky integrate-and-fire neuron."""
    
    def __init__(
        self,
        neuron_id: int,
        params: Optional[LIFParameters] = None,
        dt: float = 0.1,
    ):
        """
        Initialize LIF neuron.
        
        Args:
            neuron_id: Unique neuron identifier (bodyId from connectome)
            params: LIFParameters instance (uses defaults if None)
            dt: Integration timestep (ms)
        """
        self.neuron_id = neuron_id
        self.params = params or LIFParameters.drosophila_defaults()
        self.dt = dt
        
        # State variables
        self.V = self.params.E_leak  # Membrane potential
        self.I_syn = 0.0  # Synaptic current
        self.I_ext = 0.0  # External current
        
        # Refractory state
        self.last_spike_time = -np.inf  # Time of last spike
        self.in_refractory = False
        
        # Recording
        self.spike_times = []
        self.V_history = []
        self.I_history = []
    
    def reset(self) -> None:
        """Reset neuron to initial state."""
        self.V = self.params.E_leak
        self.I_syn = 0.0
        self.I_ext = 0.0
        self.last_spike_time = -np.inf
        self.in_refractory = False
        self.spike_times = []
        self.V_history = []
        self.I_history = []
    
    def set_synaptic_current(self, I_syn: float) -> None:
        """Set current synaptic input current (pA)."""
        self.I_syn = I_syn
    
    def set_external_current(self, I_ext: float) -> None:
        """Set external input current (pA)."""
        self.I_ext = I_ext
    
    def step(self, time: float) -> bool:
        """
        Execute one integration step.
        
        Args:
            time: Current simulation time (ms)
            
        Returns:
            True if neuron spiked, False otherwise
        """
        # Check refractory period
        if time - self.last_spike_time < self.params.tau_ref:
            self.in_refractory = True
            # During refractory period, hold at reset potential
            self.V = self.params.V_reset
            return False
        else:
            self.in_refractory = False
        
        # Add noise
        noise = np.random.normal(0, self.params.sigma_noise) if self.params.sigma_noise > 0 else 0
        
        # Total input current (pA converted to nA for integration)
        I_total = (self.I_syn + self.I_ext + noise) / 1000.0  # Convert pA to nA
        
        # LIF equation: tau_m * dV/dt = (E_leak - V) + R * I_total
        # where R is input resistance (≈ 1/g_leak in standard units)
        # Using Euler method for integration
        R = 1.0 / self.params.g_leak if self.params.g_leak > 0 else 1.0
        dV = (self.params.E_leak - self.V + R * I_total) / self.params.tau_m
        
        self.V += dV * self.dt
        
        # Clip voltage to prevent unrealistic values
        self.V = np.clip(self.V, -150, 100)
        
        # Check for spike
        spiked = False
        if self.V > self.params.V_th:
            spiked = True
            self.spike_times.append(time)
            self.last_spike_time = time
            self.V = self.params.V_reset
        
        return spiked
    
    def record(self) -> None:
        """Record current state to history."""
        self.V_history.append(self.V)
        self.I_history.append(self.I_syn + self.I_ext)
    
    def get_firing_rate(self, start_time: float = 0.0, end_time: Optional[float] = None) -> float:
        """
        Calculate average firing rate.
        
        Args:
            start_time: Start of analysis window (ms)
            end_time: End of analysis window (ms)
            
        Returns:
            Firing rate in Hz
        """
        if end_time is None:
            end_time = self.spike_times[-1] if self.spike_times else 0
        
        spikes_in_window = sum(
            1 for t in self.spike_times 
            if start_time <= t <= end_time
        )
        
        duration = (end_time - start_time) / 1000.0  # Convert ms to seconds
        if duration <= 0:
            return 0.0
        
        return spikes_in_window / duration
    
    def get_interspike_intervals(self) -> np.ndarray:
        """
        Get interspike intervals (ISI).
        
        Returns:
            Array of ISIs in milliseconds
        """
        if len(self.spike_times) < 2:
            return np.array([])
        
        return np.diff(self.spike_times)
    
    def __repr__(self) -> str:
        return f"LIFNeuron(id={self.neuron_id}, V={self.V:.2f}mV, spikes={len(self.spike_times)})"


class LIFNeuronPopulation:
    """Population of LIF neurons with efficient vectorized operations."""
    
    def __init__(
        self,
        neuron_ids: np.ndarray,
        params: Optional[LIFParameters] = None,
        dt: float = 0.1,
    ):
        """
        Initialize neuron population.
        
        Args:
            neuron_ids: Array of neuron identifiers
            params: LIFParameters (applied to all neurons)
            dt: Integration timestep (ms)
        """
        self.neuron_ids = neuron_ids
        self.params = params or LIFParameters.drosophila_defaults()
        self.dt = dt
        self.n_neurons = len(neuron_ids)
        
        # Vectorized state
        self.V = np.full(self.n_neurons, self.params.E_leak, dtype=np.float32)
        self.I_syn = np.zeros(self.n_neurons, dtype=np.float32)
        self.I_ext = np.zeros(self.n_neurons, dtype=np.float32)
        self.last_spike_time = np.full(self.n_neurons, -np.inf, dtype=np.float32)
        
        # Spike recording
        self.spike_times = [[] for _ in range(self.n_neurons)]
        
        # Create mapping from bodyId to index
        self.id_to_idx = {bid: i for i, bid in enumerate(neuron_ids)}
    
    def reset(self) -> None:
        """Reset all neurons."""
        self.V[:] = self.params.E_leak
        self.I_syn[:] = 0.0
        self.I_ext[:] = 0.0
        self.last_spike_time[:] = -np.inf
        self.spike_times = [[] for _ in range(self.n_neurons)]
    
    def set_synaptic_current(self, I_syn: np.ndarray) -> None:
        """Set synaptic current for all neurons (vectorized)."""
        self.I_syn[:] = I_syn
    
    def set_external_current(self, I_ext: np.ndarray) -> None:
        """Set external current for all neurons (vectorized)."""
        self.I_ext[:] = I_ext
    
    def step(self, time: float) -> np.ndarray:
        """
        Execute one integration step for all neurons.
        
        Args:
            time: Current simulation time (ms)
            
        Returns:
            Boolean array indicating which neurons spiked
        """
        # Refractory period mask
        in_ref = (time - self.last_spike_time) < self.params.tau_ref
        
        # Update voltage (but not for refractory neurons)
        noise = np.random.normal(0, self.params.sigma_noise, self.n_neurons) \
            if self.params.sigma_noise > 0 else np.zeros(self.n_neurons)
        
        I_total = (self.I_syn + self.I_ext + noise) / 1000.0
        R = 1.0 / self.params.g_leak if self.params.g_leak > 0 else 1.0
        dV = (self.params.E_leak - self.V + R * I_total) / self.params.tau_m
        
        self.V[~in_ref] += dV[~in_ref] * self.dt
        self.V[in_ref] = self.params.V_reset  # Hold refractory at reset
        
        # Clip to prevent unrealistic values
        self.V[:] = np.clip(self.V, -150, 100)
        
        # Detect spikes
        spiked = self.V > self.params.V_th
        
        # Record spike times
        spike_indices = np.where(spiked)[0]
        for idx in spike_indices:
            self.spike_times[idx].append(time)
            self.last_spike_time[idx] = time
        
        # Reset spiked neurons
        self.V[spiked] = self.params.V_reset
        
        return spiked
    
    def get_firing_rates(self, start_time: float = 0.0, end_time: Optional[float] = None) -> np.ndarray:
        """Get firing rates for all neurons."""
        if end_time is None:
            all_spikes = [t for spike_list in self.spike_times for t in spike_list]
            end_time = max(all_spikes) if all_spikes else 0
        
        rates = np.zeros(self.n_neurons, dtype=np.float32)
        duration = (end_time - start_time) / 1000.0  # Convert to seconds
        
        if duration > 0:
            for i in range(self.n_neurons):
                spikes_in_window = sum(
                    1 for t in self.spike_times[i]
                    if start_time <= t <= end_time
                )
                rates[i] = spikes_in_window / duration
        
        return rates
    
    def __len__(self) -> int:
        return self.n_neurons
    
    def __repr__(self) -> str:
        avg_rate = np.mean(self.get_firing_rates())
        return f"LIFPopulation(n={self.n_neurons}, avg_rate={avg_rate:.2f}Hz)"


if __name__ == "__main__":
    # Test single neuron
    params = LIFParameters.drosophila_defaults()
    neuron = LIFNeuron(neuron_id=12345, params=params)
    
    # Simulate with step current injection
    for t in np.arange(0, 100, 0.1):
        # Step current: 0-50ms off, 50-100ms on
        I_ext = 100.0 if t > 50 else 0.0
        neuron.set_external_current(I_ext)
        neuron.step(t)
    
    print(f"Neuron fired {len(neuron.spike_times)} spikes")
    if len(neuron.spike_times) > 1:
        print(f"ISI mean: {np.mean(neuron.get_interspike_intervals()):.2f} ms")
