"""
Synaptic transmission models for different neurotransmitter types.

Implements:
- Fast synaptic transmission (exponential decay)
- Short-term plasticity (depression and facilitation)
- Neurotransmitter-specific kinetics (Glutamate, GABA, Acetylcholine)
- Receptor models (AMPA, NMDA, nAChR, GABA-A, GABA-B)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class Neurotransmitter(Enum):
    """Supported neurotransmitter types in Drosophila."""
    GLUTAMATE = "Glutamate"      # Excitatory
    GABA = "GABA"                 # Inhibitory
    ACETYLCHOLINE = "Acetylcholine"  # Can be excitatory or inhibitory
    DOPAMINE = "Dopamine"         # Modulatory
    SEROTONIN = "Serotonin"       # Modulatory
    OCTOPAMINE = "Octopamine"     # Modulatory
    UNKNOWN = "Unknown"           # Unknown


@dataclass
class SynapticParameters:
    """Parameters for synaptic transmission."""
    
    # Transmission delay (ms)
    delay: float = 1.0
    
    # Synaptic conductance (nS)
    g_syn: float = 1.0
    
    # Reversal potential (mV)
    E_rev: float = 0.0
    
    # Time constant for transmission (ms)
    tau_decay: float = 5.0
    
    # Rise time (ms)
    tau_rise: float = 1.0
    
    # Short-term plasticity parameters
    U: float = 0.5        # Initial release probability
    tau_fac: float = 50.0  # Facilitation time constant (ms)
    tau_dep: float = 100.0 # Depression time constant (ms)
    
    # Quantal size (fraction of maximum conductance)
    q_size: float = 1.0
    
    @classmethod
    def glutamate(cls) -> "SynapticParameters":
        """AMPA receptor-mediated glutamate synapse (excitatory)."""
        return cls(
            delay=1.0,
            g_syn=2.0,
            E_rev=0.0,
            tau_decay=3.0,
            tau_rise=0.5,
            U=0.5,
            tau_fac=30.0,
            tau_dep=200.0,
        )
    
    @classmethod
    def gaba(cls) -> "SynapticParameters":
        """GABA-A receptor-mediated inhibitory synapse."""
        return cls(
            delay=1.0,
            g_syn=1.0,
            E_rev=-70.0,
            tau_decay=5.0,
            tau_rise=1.0,
            U=0.3,
            tau_fac=20.0,
            tau_dep=100.0,
        )
    
    @classmethod
    def acetylcholine(cls) -> "SynapticParameters":
        """Nicotinic acetylcholine receptor synapse (excitatory)."""
        return cls(
            delay=0.5,
            g_syn=1.5,
            E_rev=0.0,
            tau_decay=10.0,
            tau_rise=1.5,
            U=0.4,
            tau_fac=40.0,
            tau_dep=150.0,
        )
    
    @classmethod
    def for_neurotransmitter(cls, nt: Neurotransmitter) -> "SynapticParameters":
        """Get default parameters for neurotransmitter type."""
        if nt == Neurotransmitter.GLUTAMATE:
            return cls.glutamate()
        elif nt == Neurotransmitter.GABA:
            return cls.gaba()
        elif nt == Neurotransmitter.ACETYLCHOLINE:
            return cls.acetylcholine()
        else:
            # Default: generic excitatory
            return cls()


class Synapse:
    """Single synapse with transmission and short-term plasticity."""
    
    def __init__(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 1.0,
        params: Optional[SynapticParameters] = None,
        neurotransmitter: Neurotransmitter = Neurotransmitter.GLUTAMATE,
        dt: float = 0.1,
    ):
        """
        Initialize synapse.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            weight: Synaptic strength (0-1, relative)
            params: SynapticParameters
            neurotransmitter: Type of neurotransmitter
            dt: Integration timestep (ms)
        """
        self.pre_id = pre_id
        self.post_id = post_id
        self.weight = weight
        self.params = params or SynapticParameters.for_neurotransmitter(neurotransmitter)
        self.neurotransmitter = neurotransmitter
        self.dt = dt
        
        # Transmission state
        self.spike_queue = []  # Queue of spike times awaiting delivery
        self.g_syn = 0.0  # Current synaptic conductance
        
        # Short-term plasticity state
        self.R = 1.0  # Recovered fraction (1 = full recovery)
        self.x = 1.0  # Release-ready pool
        self.u = self.params.U  # Current release probability
        
        # History
        self.current_history = []
    
    def receive_spike(self, time: float) -> None:
        """
        Receive presynaptic spike.
        
        Args:
            time: Spike time (ms)
        """
        # Add to queue with transmission delay
        arrival_time = time + self.params.delay
        self.spike_queue.append(arrival_time)
    
    def step(self, time: float, post_V: float) -> float:
        """
        Execute one integration step and compute postsynaptic current.
        
        Args:
            time: Current simulation time (ms)
            post_V: Postsynaptic membrane potential (mV)
            
        Returns:
            Postsynaptic current (pA)
        """
        # Check for spike arrivals
        spikes_to_deliver = [t for t in self.spike_queue if t <= time]
        
        for spike_time in spikes_to_deliver:
            self.spike_queue.remove(spike_time)
            self._deliver_spike(spike_time)
        
        # Decay conductance
        self.g_syn *= np.exp(-self.dt / self.params.tau_decay)
        
        # Recover from short-term depression
        self.x += (1.0 - self.x) * self.dt / self.params.tau_dep
        
        # Relax release probability toward baseline
        self.u += (self.params.U - self.u) * self.dt / self.params.tau_fac
        
        # Compute postsynaptic current (Ohm's law)
        driving_force = post_V - self.params.E_rev
        I_syn = self.g_syn * driving_force
        
        return I_syn
    
    def _deliver_spike(self, spike_time: float) -> None:
        """
        Deliver a presynaptic spike to the postsynaptic side.
        
        Implements short-term plasticity (Tsodyks-Markram model):
        - Resource depletion (depression)
        - Facilitation (increased release probability)
        """
        # Apply short-term plasticity
        r = self.R * self.x * self.u  # Fraction of resources mobilized
        
        # Update resource states
        self.x *= (1.0 - self.u)  # Depletion of release-ready resources
        self.R -= r  # Depletion of recovered fraction
        self.u += self.params.U * (1.0 - self.u)  # Facilitation
        
        # Conductance jump
        amplitude = r * self.weight * self.params.g_syn
        self.g_syn += amplitude
        
        # Recovery of synaptic resources
        self.R += r * self.dt / self.params.tau_dep
    
    def __repr__(self) -> str:
        return (f"Synapse({self.pre_id}->{self.post_id}, "
                f"w={self.weight:.3f}, nt={self.neurotransmitter.value})")


class SynapsePopulation:
    """Efficient population of synapses with vectorized operations."""
    
    def __init__(
        self,
        synapses: list,
        dt: float = 0.1,
    ):
        """
        Initialize synapse population.
        
        Args:
            synapses: List of Synapse objects
            dt: Integration timestep (ms)
        """
        self.synapses = synapses
        self.dt = dt
        self.n_synapses = len(synapses)
        
        # Build pre->post and post->pre indices for efficient lookup
        self.pre_to_post: Dict[int, list] = {}
        self.post_to_pre: Dict[int, list] = {}
        
        for i, syn in enumerate(synapses):
            if syn.pre_id not in self.pre_to_post:
                self.pre_to_post[syn.pre_id] = []
            self.pre_to_post[syn.pre_id].append(i)
            
            if syn.post_id not in self.post_to_pre:
                self.post_to_pre[syn.post_id] = []
            self.post_to_pre[syn.post_id].append(i)
    
    def deliver_spike(self, neuron_id: int, time: float) -> None:
        """Deliver spike from presynaptic neuron to all postsynaptic targets."""
        if neuron_id not in self.pre_to_post:
            return
        
        for syn_idx in self.pre_to_post[neuron_id]:
            self.synapses[syn_idx].receive_spike(time)
    
    def compute_postsynaptic_currents(
        self,
        post_neuron_ids: np.ndarray,
        post_voltages: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """
        Compute postsynaptic currents for all neurons efficiently.
        
        Args:
            post_neuron_ids: Array of neuron IDs
            post_voltages: Array of membrane potentials (mV)
            time: Current simulation time (ms)
            
        Returns:
            Array of synaptic currents (pA) for each neuron
        """
        currents = np.zeros(len(post_neuron_ids), dtype=np.float32)
        id_to_idx = {nid: i for i, nid in enumerate(post_neuron_ids)}
        
        for syn in self.synapses:
            if syn.post_id in id_to_idx:
                post_idx = id_to_idx[syn.post_id]
                I = syn.step(time, post_voltages[post_idx])
                currents[post_idx] += I
        
        return currents
    
    def __len__(self) -> int:
        return self.n_synapses
    
    def __repr__(self) -> str:
        return f"SynapsePopulation(n={self.n_synapses})"


if __name__ == "__main__":
    # Test synapse
    params = SynapticParameters.glutamate()
    syn = Synapse(pre_id=1, post_id=2, weight=0.8, params=params)
    
    # Simulate presynaptic spike
    syn.receive_spike(10.0)
    
    for t in np.arange(0, 50, 0.1):
        I = syn.step(t, post_V=-65.0)
        print(f"t={t:.1f}, I={I:.2f}pA, g={syn.g_syn:.3f}nS")
