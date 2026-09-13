"""
Visualization and plotting utilities for network simulation results.

Creates:
- Spike raster plots
- Population activity curves
- Firing rate distributions
- Network connectivity diagrams
- Phase space trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SimulationPlotter:
    """Plotting utilities for simulation results."""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10), dpi: int = 100):
        """
        Initialize plotter.
        
        Args:
            figsize: Figure size (width, height)
            dpi: Dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("darkgrid")
    
    def plot_raster(
        self,
        spike_times: Dict[int, list],
        duration: float,
        max_neurons: int = 1000,
        figsize: Optional[Tuple] = None,
        title: str = "Spike Raster",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot spike raster diagram.
        
        Args:
            spike_times: Dictionary mapping neuron_id to spike times
            duration: Simulation duration (ms)
            max_neurons: Maximum neurons to plot (samples if more)
            figsize: Figure size override
            title: Plot title
            
        Returns:
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        # Sample neurons if too many
        neuron_ids = list(spike_times.keys())
        if len(neuron_ids) > max_neurons:
            neuron_ids = np.random.choice(neuron_ids, size=max_neurons, replace=False)
        
        # Plot spikes
        for idx, neuron_id in enumerate(neuron_ids):
            times = spike_times[neuron_id]
            if times:
                ax.vlines(times, idx - 0.5, idx + 0.5, colors='black', linewidth=0.5)
        
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel(f"Neuron Index (of {len(neuron_ids)})", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, duration)
        ax.set_ylim(-1, len(neuron_ids))
        
        fig.tight_layout()
        return fig, ax
    
    def plot_population_activity(
        self,
        spike_times: Dict[int, list],
        duration: float,
        window_size: float = 10.0,
        figsize: Optional[Tuple] = None,
        title: str = "Population Activity",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot population spike count over time.
        
        Args:
            spike_times: Dictionary mapping neuron_id to spike times
            duration: Simulation duration (ms)
            window_size: Bin size for smoothing (ms)
            figsize: Figure size override
            title: Plot title
            
        Returns:
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize or (12, 5), dpi=self.dpi)
        
        # Create time bins
        n_bins = int(duration / window_size)
        bin_edges = np.linspace(0, duration, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Count spikes per bin
        spike_counts = np.zeros(n_bins)
        for times in spike_times.values():
            hist, _ = np.histogram(times, bins=bin_edges)
            spike_counts += hist
        
        # Convert to Hz
        n_neurons = len(spike_times)
        if n_neurons > 0:
            firing_rate = spike_counts / (n_neurons * (window_size / 1000.0))
        else:
            firing_rate = np.zeros_like(spike_counts)
        
        ax.fill_between(bin_centers, firing_rate, alpha=0.5, color='blue')
        ax.plot(bin_centers, firing_rate, color='darkblue', linewidth=2)
        
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Population Firing Rate (Hz)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig, ax
    
    def plot_firing_rate_distribution(
        self,
        spike_times: Dict[int, list],
        duration: float,
        figsize: Optional[Tuple] = None,
        title: str = "Firing Rate Distribution",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot histogram of neuron firing rates.
        
        Args:
            spike_times: Dictionary mapping neuron_id to spike times
            duration: Simulation duration (ms)
            figsize: Figure size override
            title: Plot title
            
        Returns:
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize or (10, 6), dpi=self.dpi)
        
        # Compute firing rates
        firing_rates = []
        duration_sec = duration / 1000.0
        
        for times in spike_times.values():
            if duration_sec > 0:
                firing_rates.append(len(times) / duration_sec)
        
        if not firing_rates:
            logger.warning("No spike data to plot")
            return fig, ax
        
        # Plot histogram
        ax.hist(firing_rates, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(firing_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(firing_rates):.1f} Hz')
        ax.axvline(np.median(firing_rates), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(firing_rates):.1f} Hz')
        
        ax.set_xlabel("Firing Rate (Hz)", fontsize=12)
        ax.set_ylabel("Number of Neurons", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        return fig, ax
    
    def plot_interspike_intervals(
        self,
        spike_times: Dict[int, list],
        figsize: Optional[Tuple] = None,
        title: str = "Interspike Interval Distribution",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot histogram of interspike intervals.
        
        Args:
            spike_times: Dictionary mapping neuron_id to spike times
            figsize: Figure size override
            title: Plot title
            
        Returns:
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize or (10, 6), dpi=self.dpi)
        
        # Compute ISIs
        isis = []
        for times in spike_times.values():
            if len(times) > 1:
                isis.extend(np.diff(times))
        
        if not isis:
            logger.warning("No ISI data to plot")
            return fig, ax
        
        # Plot histogram (log scale)
        ax.hist(isis, bins=100, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Interspike Interval (ms)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        return fig, ax
    
    def plot_voltage_trace(
        self,
        voltage_history: Dict[int, list],
        neuron_indices: Optional[list] = None,
        dt: float = 0.1,
        figsize: Optional[Tuple] = None,
        title: str = "Membrane Potential Traces",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot membrane potential over time for selected neurons.
        
        Args:
            voltage_history: Dictionary mapping neuron_id to voltage traces
            neuron_indices: Indices to plot (sample if None)
            dt: Timestep (ms)
            figsize: Figure size override
            title: Plot title
            
        Returns:
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize or (12, 6), dpi=self.dpi)
        
        neuron_ids = list(voltage_history.keys())
        if neuron_indices is None:
            neuron_indices = list(range(min(5, len(neuron_ids))))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_indices)))
        
        for idx, color in zip(neuron_indices, colors):
            if idx < len(neuron_ids):
                neuron_id = neuron_ids[idx]
                voltages = voltage_history[neuron_id]
                time = np.arange(len(voltages)) * dt
                ax.plot(time, voltages, label=f"Neuron {idx}", color=color, linewidth=1)
        
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Membrane Potential (mV)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig, ax
    
    def plot_network_summary(
        self,
        spike_times: Dict[int, list],
        duration: float,
        window_size: float = 10.0,
        figsize: Optional[Tuple] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create multi-panel summary figure.
        
        Args:
            spike_times: Dictionary mapping neuron_id to spike times
            duration: Simulation duration (ms)
            window_size: Bin size for population activity (ms)
            figsize: Figure size override
            
        Returns:
            (figure, axes_array)
        """
        fig = plt.figure(figsize=figsize or (16, 12), dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Raster plot
        ax1 = fig.add_subplot(gs[0, :])
        self._add_raster_to_ax(ax1, spike_times, duration, max_neurons=500)
        ax1.set_title("Spike Raster (sample)", fontsize=12)
        
        # Population activity
        ax2 = fig.add_subplot(gs[1, 0])
        self._add_population_activity_to_ax(ax2, spike_times, duration, window_size)
        ax2.set_title("Population Activity", fontsize=12)
        
        # Firing rate distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._add_firing_rate_dist_to_ax(ax3, spike_times, duration)
        ax3.set_title("Firing Rate Distribution", fontsize=12)
        
        # ISI distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self._add_isi_dist_to_ax(ax4, spike_times)
        ax4.set_title("Interspike Interval Distribution", fontsize=12)
        
        # Statistics text
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        stats_text = self._get_statistics_text(spike_times, duration)
        ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        return fig, np.array([ax1, ax2, ax3, ax4, ax5])
    
    def _add_raster_to_ax(
        self,
        ax,
        spike_times: Dict[int, list],
        duration: float,
        max_neurons: int = 500,
    ) -> None:
        """Add raster plot to existing axes."""
        neuron_ids = list(spike_times.keys())
        if len(neuron_ids) > max_neurons:
            neuron_ids = np.random.choice(neuron_ids, size=max_neurons, replace=False)
        
        for idx, neuron_id in enumerate(neuron_ids):
            times = spike_times[neuron_id]
            if times:
                ax.vlines(times, idx - 0.5, idx + 0.5, colors='black', linewidth=0.5)
        
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron")
        ax.set_xlim(0, duration)
        ax.set_ylim(-1, len(neuron_ids))
    
    def _add_population_activity_to_ax(
        self,
        ax,
        spike_times: Dict[int, list],
        duration: float,
        window_size: float,
    ) -> None:
        """Add population activity to existing axes."""
        n_bins = int(duration / window_size)
        bin_edges = np.linspace(0, duration, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        spike_counts = np.zeros(n_bins)
        for times in spike_times.values():
            hist, _ = np.histogram(times, bins=bin_edges)
            spike_counts += hist
        
        n_neurons = len(spike_times)
        if n_neurons > 0:
            firing_rate = spike_counts / (n_neurons * (window_size / 1000.0))
        else:
            firing_rate = np.zeros_like(spike_counts)
        
        ax.fill_between(bin_centers, firing_rate, alpha=0.5, color='blue')
        ax.plot(bin_centers, firing_rate, color='darkblue', linewidth=2)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.3)
    
    def _add_firing_rate_dist_to_ax(
        self,
        ax,
        spike_times: Dict[int, list],
        duration: float,
    ) -> None:
        """Add firing rate distribution to existing axes."""
        firing_rates = []
        duration_sec = duration / 1000.0
        
        for times in spike_times.values():
            if duration_sec > 0:
                firing_rates.append(len(times) / duration_sec)
        
        if firing_rates:
            ax.hist(firing_rates, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(firing_rates), color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel("Firing Rate (Hz)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3, axis='y')
    
    def _add_isi_dist_to_ax(
        self,
        ax,
        spike_times: Dict[int, list],
    ) -> None:
        """Add ISI distribution to existing axes."""
        isis = []
        for times in spike_times.values():
            if len(times) > 1:
                isis.extend(np.diff(times))
        
        if isis:
            ax.hist(isis, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax.set_yscale("log")
        
        ax.set_xlabel("Interspike Interval (ms)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3, axis='y')
    
    def _get_statistics_text(
        self,
        spike_times: Dict[int, list],
        duration: float,
    ) -> str:
        """Generate statistics text."""
        duration_sec = duration / 1000.0
        
        firing_rates = []
        total_spikes = 0
        
        for times in spike_times.values():
            n_spikes = len(times)
            total_spikes += n_spikes
            if duration_sec > 0:
                firing_rates.append(n_spikes / duration_sec)
        
        active_neurons = sum(1 for times in spike_times.values() if len(times) > 0)
        
        text = f"""Simulation Statistics
Duration: {duration:.1f} ms
Total neurons: {len(spike_times)}
Active neurons: {active_neurons}

Spike Statistics
Total spikes: {total_spikes}
Mean firing rate: {np.mean(firing_rates):.2f} Hz
Max firing rate: {np.max(firing_rates):.2f} Hz
Min firing rate: {np.min(firing_rates):.2f} Hz
"""
        
        return text


if __name__ == "__main__":
    logger.info("Visualization module for LIF simulation")
    logger.info("Use: plotter = SimulationPlotter()")
