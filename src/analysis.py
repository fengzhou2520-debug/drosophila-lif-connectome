"""
Network analysis and statistics computation.

Analyzes simulation results including:
- Firing rate statistics
- Spike timing and correlation analysis
- Network activity patterns
- Connectivity analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.signal import correlate
import logging

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """Analyze network simulation results."""
    
    def __init__(self, network, simulation_results: Dict):
        """
        Initialize analyzer.
        
        Args:
            network: FlyBrainNetwork instance
            simulation_results: Results from LIFSimulator.run()
        """
        self.network = network
        self.results = simulation_results
        self.spike_times = simulation_results.get("spike_times", {})
        self.duration = simulation_results.get("duration", 0)
        self.statistics = {}
    
    def compute_firing_rates(self) -> Dict[int, float]:
        """Compute average firing rate for each neuron."""
        firing_rates = {}
        duration_sec = self.duration / 1000.0
        
        if duration_sec <= 0:
            return firing_rates
        
        for neuron_id, times in self.spike_times.items():
            firing_rates[neuron_id] = len(times) / duration_sec
        
        self.statistics["firing_rates"] = firing_rates
        return firing_rates
    
    def compute_firing_rate_distribution(self) -> Dict:
        """Get statistics of firing rate distribution."""
        if "firing_rates" not in self.statistics:
            self.compute_firing_rates()
        
        rates = list(self.statistics["firing_rates"].values())
        
        if not rates:
            return {}
        
        return {
            "mean": np.mean(rates),
            "median": np.median(rates),
            "std": np.std(rates),
            "min": np.min(rates),
            "max": np.max(rates),
            "q1": np.percentile(rates, 25),
            "q3": np.percentile(rates, 75),
        }
    
    def compute_interspike_intervals(self) -> Dict[int, np.ndarray]:
        """Compute interspike intervals (ISI) for each neuron."""
        isis = {}
        
        for neuron_id, times in self.spike_times.items():
            if len(times) > 1:
                isis[neuron_id] = np.diff(times)
        
        self.statistics["isis"] = isis
        return isis
    
    def compute_coefficient_of_variation(self) -> Dict[int, float]:
        """Compute coefficient of variation (CV) of ISI for each neuron."""
        if "isis" not in self.statistics:
            self.compute_interspike_intervals()
        
        cv = {}
        for neuron_id, isi in self.statistics["isis"].items():
            if len(isi) > 0 and np.mean(isi) > 0:
                cv[neuron_id] = np.std(isi) / np.mean(isi)
        
        self.statistics["cv"] = cv
        return cv
    
    def compute_pairwise_correlations(self, time_window: float = 10.0) -> Dict:
        """
        Compute pairwise spike train correlations.
        
        Args:
            time_window: Correlation window (ms)
            
        Returns:
            Dictionary with correlation statistics
        """
        spike_times_list = list(self.spike_times.values())
        
        if len(spike_times_list) < 2:
            logger.warning("Insufficient neurons for correlation analysis")
            return {}
        
        # Convert spike times to binary vectors
        n_bins = int(self.duration / time_window)
        
        correlations = []
        
        for i in range(min(len(spike_times_list), 100)):  # Sample 100 neurons
            spike_train_i = np.zeros(n_bins)
            for spike_time in spike_times_list[i]:
                bin_idx = int(spike_time / time_window)
                if 0 <= bin_idx < n_bins:
                    spike_train_i[bin_idx] += 1
            
            for j in range(i + 1, min(len(spike_times_list), 100)):
                spike_train_j = np.zeros(n_bins)
                for spike_time in spike_times_list[j]:
                    bin_idx = int(spike_time / time_window)
                    if 0 <= bin_idx < n_bins:
                        spike_train_j[bin_idx] += 1
                
                # Compute correlation
                if np.std(spike_train_i) > 0 and np.std(spike_train_j) > 0:
                    corr = np.corrcoef(spike_train_i, spike_train_j)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if correlations:
            return {
                "mean_correlation": np.mean(correlations),
                "std_correlation": np.std(correlations),
                "n_pairs": len(correlations),
            }
        else:
            return {}
    
    def compute_synchrony(self, time_window: float = 5.0) -> np.ndarray:
        """
        Compute population synchrony over time.
        
        Args:
            time_window: Time bin size (ms)
            
        Returns:
            Array of synchrony values
        """
        n_bins = int(self.duration / time_window)
        spike_counts = np.zeros(n_bins)
        
        for times in self.spike_times.values():
            for spike_time in times:
                bin_idx = int(spike_time / time_window)
                if 0 <= bin_idx < n_bins:
                    spike_counts[bin_idx] += 1
        
        # Normalize by number of neurons
        n_neurons = len(self.spike_times)
        if n_neurons > 0:
            spike_counts /= n_neurons
        
        return spike_counts
    
    def compute_burst_statistics(self, isi_threshold: float = 20.0) -> Dict:
        """
        Compute burst statistics.
        
        Args:
            isi_threshold: ISI threshold for burst detection (ms)
            
        Returns:
            Dictionary with burst statistics
        """
        if "isis" not in self.statistics:
            self.compute_interspike_intervals()
        
        burst_stats = {
            "n_bursting_neurons": 0,
            "mean_burst_length": 0,
            "mean_burst_frequency": 0,
        }
        
        burst_lengths = []
        burst_frequencies = []
        
        for neuron_id, isi in self.statistics["isis"].items():
            if len(isi) < 2:
                continue
            
            # Identify bursts (ISI < threshold)
            in_burst = isi < isi_threshold
            burst_groups = np.split(np.arange(len(isi)), np.where(np.diff(in_burst))[0] + 1)
            
            bursts = [g for g, mask in zip(burst_groups, 
                                           [in_burst[g[0]] if len(g) > 0 else False 
                                            for g in burst_groups]) if mask and len(g) > 1]
            
            if bursts:
                burst_stats["n_bursting_neurons"] += 1
                burst_lengths.extend([len(b) + 1 for b in bursts])
                burst_frequencies.append(len(bursts) / (self.duration / 1000.0))
        
        if burst_lengths:
            burst_stats["mean_burst_length"] = np.mean(burst_lengths)
        
        if burst_frequencies:
            burst_stats["mean_burst_frequency"] = np.mean(burst_frequencies)
        
        return burst_stats
    
    def get_active_neurons(self) -> int:
        """Get number of neurons that fired at least one spike."""
        return sum(1 for times in self.spike_times.values() if len(times) > 0)
    
    def get_silent_neurons(self) -> int:
        """Get number of neurons that did not fire."""
        return sum(1 for times in self.spike_times.values() if len(times) == 0)
    
    def summary(self) -> str:
        """Generate summary report."""
        self.compute_firing_rates()
        self.compute_interspike_intervals()
        self.compute_coefficient_of_variation()
        
        fire_rate_dist = self.compute_firing_rate_distribution()
        burst_stats = self.compute_burst_statistics()
        
        active = self.get_active_neurons()
        silent = self.get_silent_neurons()
        total = len(self.spike_times)
        
        report = f"""
Network Analysis Summary
{'=' * 50}
Duration: {self.duration:.1f} ms
Total neurons: {total}
Active neurons: {active} ({100*active/total:.1f}%)
Silent neurons: {silent} ({100*silent/total:.1f}%)

Firing Rate Statistics (Hz)
  Mean: {fire_rate_dist.get('mean', 0):.2f}
  Median: {fire_rate_dist.get('median', 0):.2f}
  Std: {fire_rate_dist.get('std', 0):.2f}
  Max: {fire_rate_dist.get('max', 0):.2f}

Spike Timing
  Mean CV (ISI): {np.mean(list(self.statistics.get('cv', {}).values())) if self.statistics.get('cv') else 0:.2f}

Bursting
  Bursting neurons: {burst_stats.get('n_bursting_neurons', 0)}
  Mean burst length: {burst_stats.get('mean_burst_length', 0):.1f} spikes
  Mean burst frequency: {burst_stats.get('mean_burst_frequency', 0):.2f} bursts/sec

Total spikes: {sum(len(times) for times in self.spike_times.values())}
{'=' * 50}
        """
        
        return report


class ConnectivityAnalyzer:
    """Analyze network connectivity structure."""
    
    def __init__(self, network):
        """Initialize connectivity analyzer."""
        self.network = network
    
    def get_degree_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get in-degree and out-degree distributions."""
        in_degree = {}
        out_degree = {}
        
        for syn in self.network.synapse_list:
            out_degree[syn.pre_id] = out_degree.get(syn.pre_id, 0) + 1
            in_degree[syn.post_id] = in_degree.get(syn.post_id, 0) + 1
        
        in_degrees = np.array(list(in_degree.values())) if in_degree else np.array([])
        out_degrees = np.array(list(out_degree.values())) if out_degree else np.array([])
        
        return in_degrees, out_degrees
    
    def get_strongly_connected_neurons(self) -> Dict[int, int]:
        """Get hub neurons (high in-degree and out-degree)."""
        in_deg, out_deg = self.get_degree_distribution()
        
        hubs = {}
        for syn in self.network.synapse_list:
            total_degree = (self.network.synapses.pre_to_post.get(syn.pre_id, []) +
                           self.network.synapses.post_to_pre.get(syn.pre_id, []))
            if len(total_degree) > np.percentile(in_deg, 90):
                hubs[syn.pre_id] = len(total_degree)
        
        return hubs
    
    def get_motif_statistics(self) -> Dict:
        """Analyze common connectivity motifs."""
        # Simple motif detection (feedforward, feedback, etc.)
        motifs = {
            "feedforward": 0,
            "feedback": 0,
            "divergence": 0,
            "convergence": 0,
        }
        
        # Check for reciprocal connections
        connections = set()
        for syn in self.network.synapse_list:
            connections.add((syn.pre_id, syn.post_id))
        
        for (pre1, post1) in connections:
            # Feedback
            if (post1, pre1) in connections:
                motifs["feedback"] += 1
            
            # Look for triads
            for syn2 in self.network.synapse_list:
                if syn2.pre_id == post1:
                    pre2, post2 = syn2.pre_id, syn2.post_id
                    
                    # Feedforward
                    if (pre1, post2) in connections:
                        motifs["feedforward"] += 1
                    
                    # Divergence (A->B, A->C)
                    if pre1 != pre2 and post1 == pre2:
                        motifs["divergence"] += 1
                    
                    # Convergence (A->C, B->C)
                    if post1 != post2 and pre1 != pre2 and post2 in [s.post_id for s in self.network.synapse_list 
                                                                       if s.pre_id == pre1]:
                        motifs["convergence"] += 1
        
        return motifs


if __name__ == "__main__":
    logger.info("Network analysis module")
    logger.info("Use with: analyzer = NetworkAnalyzer(network, results)")
