"""
Network construction from connectome data.

Builds a complete fly brain network from connectome, neuron, and neurotransmitter data.
Handles:
- Neuron population creation from connectome annotations
- Synapse instantiation with weights and delays
- Neurotransmitter assignment to synapses
- Network validation and statistics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

from .neuron import LIFNeuronPopulation, LIFParameters
from .synapse import SynapsePopulation, Synapse, Neurotransmitter

logger = logging.getLogger(__name__)


class Connectome:
    """Connectome graph representation."""
    
    def __init__(
        self,
        neurons_df: pd.DataFrame,
        synapses_df: pd.DataFrame,
        neurotransmitters_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize connectome.
        
        Args:
            neurons_df: DataFrame with neuron annotations (must have 'bodyId' column)
            synapses_df: DataFrame with synaptic connections (must have 'pre', 'post', 'weight' columns)
            neurotransmitters_df: Optional DataFrame mapping bodyId to neurotransmitter type
        """
        self.neurons_df = neurons_df
        self.synapses_df = synapses_df
        self.neurotransmitters_df = neurotransmitters_df
        
        # Validate
        self._validate()
        
        # Build mappings
        self.neuron_ids = self.neurons_df["bodyId"].values
        self.n_neurons = len(self.neuron_ids)
        self.id_to_idx = {bid: i for i, bid in enumerate(self.neuron_ids)}
        
        # Get neurotransmitter mapping
        self.nt_map = self._build_neurotransmitter_map()
        
        logger.info(f"Connectome: {self.n_neurons} neurons, {len(self.synapses_df)} synapses")
    
    def _validate(self) -> None:
        """Validate data integrity."""
        if "bodyId" not in self.neurons_df.columns:
            raise ValueError("neurons_df must have 'bodyId' column")
        
        if "pre" not in self.synapses_df.columns or "post" not in self.synapses_df.columns:
            raise ValueError("synapses_df must have 'pre' and 'post' columns")
        
        if "weight" not in self.synapses_df.columns:
            logger.warning("synapses_df has no 'weight' column, using weight=1.0 for all synapses")
    
    def _build_neurotransmitter_map(self) -> Dict[int, Neurotransmitter]:
        """Build mapping from bodyId to neurotransmitter type."""
        nt_map = {}
        
        if self.neurotransmitters_df is None:
            logger.warning("No neurotransmitter data provided, defaulting to GLUTAMATE")
            for bid in self.neuron_ids:
                nt_map[bid] = Neurotransmitter.GLUTAMATE
            return nt_map
        
        if "bodyId" not in self.neurotransmitters_df.columns:
            logger.warning("neurotransmitters_df missing 'bodyId' column")
            return nt_map
        
        # Map neurotransmitter strings to enum
        for _, row in self.neurotransmitters_df.iterrows():
            body_id = int(row["bodyId"])
            nt_str = row.get("neurotransmitter", "").lower()
            
            if "glutamate" in nt_str or "glu" in nt_str:
                nt = Neurotransmitter.GLUTAMATE
            elif "gaba" in nt_str:
                nt = Neurotransmitter.GABA
            elif "acetylcholine" in nt_str or "ach" in nt_str:
                nt = Neurotransmitter.ACETYLCHOLINE
            elif "dopamine" in nt_str or "da" in nt_str:
                nt = Neurotransmitter.DOPAMINE
            elif "serotonin" in nt_str or "5-ht" in nt_str:
                nt = Neurotransmitter.SEROTONIN
            elif "octopamine" in nt_str or "oa" in nt_str:
                nt = Neurotransmitter.OCTOPAMINE
            else:
                nt = Neurotransmitter.UNKNOWN
            
            nt_map[body_id] = nt
        
        logger.info(f"Mapped {len(nt_map)} neurons to neurotransmitters")
        return nt_map
    
    def get_neurotransmitter(self, body_id: int) -> Neurotransmitter:
        """Get neurotransmitter type for a neuron."""
        return self.nt_map.get(body_id, Neurotransmitter.GLUTAMATE)
    
    def get_incoming_synapses(self, body_id: int) -> pd.DataFrame:
        """Get all synapses with this neuron as postsynaptic."""
        return self.synapses_df[self.synapses_df["post"] == body_id]
    
    def get_outgoing_synapses(self, body_id: int) -> pd.DataFrame:
        """Get all synapses with this neuron as presynaptic."""
        return self.synapses_df[self.synapses_df["pre"] == body_id]
    
    def get_statistics(self) -> Dict:
        """Get connectome statistics."""
        stats = {
            "n_neurons": self.n_neurons,
            "n_synapses": len(self.synapses_df),
            "density": len(self.synapses_df) / (self.n_neurons ** 2),
        }
        
        if "weight" in self.synapses_df.columns:
            stats["weight_mean"] = self.synapses_df["weight"].mean()
            stats["weight_std"] = self.synapses_df["weight"].std()
        
        # In/out degree statistics
        in_degree = self.synapses_df["post"].value_counts()
        out_degree = self.synapses_df["pre"].value_counts()
        
        stats["in_degree_mean"] = in_degree.mean()
        stats["in_degree_max"] = in_degree.max()
        stats["out_degree_mean"] = out_degree.mean()
        stats["out_degree_max"] = out_degree.max()
        
        return stats


class FlyBrainNetwork:
    """Complete fly brain network with neurons and synapses."""
    
    def __init__(
        self,
        connectome: Connectome,
        neuron_params: Optional[LIFParameters] = None,
        dt: float = 0.1,
    ):
        """
        Initialize network.
        
        Args:
            connectome: Connectome object
            neuron_params: LIF parameters for all neurons
            dt: Integration timestep (ms)
        """
        self.connectome = connectome
        self.dt = dt
        self.neuron_params = neuron_params or LIFParameters.drosophila_defaults()
        
        # Will be populated by build()
        self.neurons = None
        self.synapses = None
        self.synapse_list = []
        
        self.built = False
    
    def build(self, verbose: bool = True) -> None:
        """
        Construct network from connectome.
        
        Creates LIF neuron population and synapse population with proper
        connectivity, weights, and neurotransmitter assignments.
        """
        logger.info("Building fly brain network...")
        
        # Create neuron population
        logger.info("Creating neuron population...")
        self.neurons = LIFNeuronPopulation(
            self.connectome.neuron_ids,
            params=self.neuron_params,
            dt=self.dt,
        )
        
        # Create synapses
        logger.info("Creating synapses from connectome...")
        synapses_data = self.connectome.synapses_df
        
        iterator = tqdm(synapses_data.iterrows(), total=len(synapses_data)) if verbose else synapses_data.iterrows()
        
        for _, row in iterator:
            pre_id = int(row["pre"])
            post_id = int(row["post"])
            weight = float(row.get("weight", 1.0))
            
            # Skip invalid connections
            if pre_id not in self.connectome.id_to_idx or post_id not in self.connectome.id_to_idx:
                continue
            
            # Get neurotransmitter type from presynaptic neuron
            nt = self.connectome.get_neurotransmitter(pre_id)
            
            # Create synapse
            syn = Synapse(
                pre_id=pre_id,
                post_id=post_id,
                weight=weight,
                neurotransmitter=nt,
                dt=self.dt,
            )
            
            self.synapse_list.append(syn)
        
        # Create synapse population
        self.synapses = SynapsePopulation(self.synapse_list, dt=self.dt)
        
        logger.info(f"Network built: {len(self.neurons)} neurons, {len(self.synapses)} synapses")
        self.built = True
        
        # Print statistics
        if verbose:
            self.print_statistics()
    
    def print_statistics(self) -> None:
        """Print network statistics."""
        logger.info("\n" + "="*60)
        logger.info("NETWORK STATISTICS")
        logger.info("="*60)
        
        stats = self.connectome.get_statistics()
        logger.info(f"Neurons: {stats['n_neurons']:,}")
        logger.info(f"Synapses: {stats['n_synapses']:,}")
        logger.info(f"Density: {stats['density']:.2e}")
        logger.info(f"Mean in-degree: {stats['in_degree_mean']:.1f}")
        logger.info(f"Max in-degree: {stats['in_degree_max']}")
        logger.info(f"Mean out-degree: {stats['out_degree_mean']:.1f}")
        logger.info(f"Max out-degree: {stats['out_degree_max']}")
        
        if "weight_mean" in stats:
            logger.info(f"Mean synapse weight: {stats['weight_mean']:.3f}")
            logger.info(f"Synapse weight std: {stats['weight_std']:.3f}")
        
        logger.info("="*60 + "\n")
    
    def get_neuron_indices(self, body_ids: List[int]) -> np.ndarray:
        """Get indices for given neuron bodyIds."""
        return np.array([self.connectome.id_to_idx[bid] for bid in body_ids])
    
    def get_connectivity_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sparse connectivity matrix.
        
        Returns:
            (indices, weights) where indices[i] = (pre, post) and weights[i] is the strength
        """
        indices = []
        weights = []
        
        for syn in self.synapse_list:
            pre_idx = self.connectome.id_to_idx.get(syn.pre_id)
            post_idx = self.connectome.id_to_idx.get(syn.post_id)
            
            if pre_idx is not None and post_idx is not None:
                indices.append((pre_idx, post_idx))
                weights.append(syn.weight)
        
        return np.array(indices), np.array(weights)
    
    def validate(self) -> bool:
        """Validate network integrity."""
        if not self.built:
            logger.error("Network not built yet")
            return False
        
        # Check connectivity
        for syn in self.synapse_list:
            if syn.pre_id not in self.connectome.id_to_idx:
                logger.error(f"Invalid presynaptic neuron: {syn.pre_id}")
                return False
            if syn.post_id not in self.connectome.id_to_idx:
                logger.error(f"Invalid postsynaptic neuron: {syn.post_id}")
                return False
        
        logger.info("✓ Network validation passed")
        return True
    
    def __repr__(self) -> str:
        status = "built" if self.built else "not built"
        return f"FlyBrainNetwork({status}, n={len(self.neurons)} neurons, m={len(self.synapses)} synapses)"


if __name__ == "__main__":
    # Example usage (requires data files)
    from .data_loader import ConnectomeDataLoader
    
    loader = ConnectomeDataLoader()
    logger.info("Loading connectome data...")
    
    try:
        loader.load_feather_files()
        connectome = Connectome(
            loader.neurons,
            loader.synapses,
            loader.neurotransmitters,
        )
        
        network = FlyBrainNetwork(connectome)
        network.build()
        network.validate()
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.info("Please download from https://male-cns.janelia.org/download/")
