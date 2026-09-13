"""
Data loader for Janelia male-CNS connectome.

Downloads and loads:
- body-annotations-male-cns-v1.0-minconf-0.5.feather (neuron metadata)
- body-neurotransmitters-male-cns-v1.0.feather (neurotransmitter types)
- connectome-weights-male-cns-v1.0-minconf-0.5.feather (synaptic connectivity)
"""

import os
import logging
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectomeDataLoader:
    """Load and manage Drosophila male-CNS connectome data from Janelia."""

    # Download URLs from Janelia's male-CNS dataset
    JANELIA_BASE_URL = "https://download.janelia.org/download"
    
    FILES = {
        "neurons": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
        "synapses": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    }

    def __init__(self, data_dir: str = "data", use_cache: bool = True):
        """
        Initialize data loader.

        Args:
            data_dir: Directory to store downloaded data
            use_cache: Use cached data if available
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.use_cache = use_cache

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.neurons = None
        self.neurotransmitters = None
        self.synapses = None
        self.neuron_types = None

    def download_janelia_data(self, force: bool = False) -> None:
        """
        Download connectome data from Janelia's server.

        Args:
            force: Force re-download even if cached
        """
        logger.info("Checking Janelia connectome data...")

        for key, filename in self.FILES.items():
            filepath = self.raw_dir / filename
            
            if filepath.exists() and not force and self.use_cache:
                logger.info(f"✓ {filename} already cached")
                continue

            logger.info(f"Downloading {filename}...")
            # Note: You'll need to manually download from:
            # https://male-cns.janelia.org/download/#__tabbed_1_1
            # This placeholder shows the download structure
            logger.warning(
                f"Please manually download {filename} from "
                "https://male-cns.janelia.org/download/#__tabbed_1_1"
            )

    def load_feather_files(self) -> None:
        """Load connectome data from cached feather files."""
        logger.info("Loading connectome data from feather files...")

        # Load neurons
        neuron_file = self.raw_dir / self.FILES["neurons"]
        if not neuron_file.exists():
            raise FileNotFoundError(
                f"Neuron file not found: {neuron_file}. "
                "Please download from https://male-cns.janelia.org/download/"
            )
        
        logger.info("Loading neuron annotations...")
        self.neurons = pd.read_feather(neuron_file)
        logger.info(f"  Loaded {len(self.neurons)} neurons")
        logger.info(f"  Columns: {list(self.neurons.columns)}")

        # Load neurotransmitters
        nt_file = self.raw_dir / self.FILES["neurotransmitters"]
        if nt_file.exists():
            logger.info("Loading neurotransmitter data...")
            self.neurotransmitters = pd.read_feather(nt_file)
            logger.info(f"  Loaded {len(self.neurotransmitters)} neurotransmitter assignments")
            logger.info(f"  Columns: {list(self.neurotransmitters.columns)}")
        else:
            logger.warning("Neurotransmitter file not found")

        # Load synapses
        synapse_file = self.raw_dir / self.FILES["synapses"]
        if not synapse_file.exists():
            raise FileNotFoundError(
                f"Synapse file not found: {synapse_file}. "
                "Please download from https://male-cns.janelia.org/download/"
            )

        logger.info("Loading synaptic connectivity...")
        self.synapses = pd.read_feather(synapse_file)
        logger.info(f"  Loaded {len(self.synapses)} synaptic connections")
        logger.info(f"  Columns: {list(self.synapses.columns)}")

    def validate_data(self) -> bool:
        """Validate loaded data integrity."""
        logger.info("Validating connectome data...")

        if self.neurons is None or self.synapses is None:
            logger.error("Data not loaded")
            return False

        # Check for required columns
        required_neuron_cols = {"bodyId"}  # Janelia uses bodyId as neuron identifier
        if not required_neuron_cols.issubset(self.neurons.columns):
            logger.error(f"Missing neuron columns: {required_neuron_cols - set(self.neurons.columns)}")
            return False

        required_synapse_cols = {"pre", "post", "weight"}  # Typical synapse table format
        if not required_synapse_cols.issubset(self.synapses.columns):
            logger.warning(f"Synapse columns: {set(self.synapses.columns)}")

        # Check for NaNs in critical columns
        if self.neurons["bodyId"].isna().any():
            logger.error("Found NaN values in neuron bodyId")
            return False

        logger.info("✓ Data validation passed")
        return True

    def get_neuron_types(self) -> Dict[str, list]:
        """Get mapping of neuron types to bodyIds."""
        if self.neuron_types is not None:
            return self.neuron_types

        logger.info("Extracting neuron types...")
        self.neuron_types = {}

        # Group by type/class if available
        if "type" in self.neurons.columns:
            for neuron_type, group in self.neurons.groupby("type"):
                self.neuron_types[str(neuron_type)] = group["bodyId"].tolist()
        elif "class" in self.neurons.columns:
            for neuron_class, group in self.neurons.groupby("class"):
                self.neuron_types[str(neuron_class)] = group["bodyId"].tolist()
        else:
            # No type information, group all as "neuron"
            self.neuron_types["generic"] = self.neurons["bodyId"].tolist()

        return self.neuron_types

    def get_neurotransmitter_map(self) -> Dict[int, str]:
        """Map neuron bodyId to primary neurotransmitter."""
        if self.neurotransmitters is None:
            logger.warning("Neurotransmitter data not loaded")
            return {}

        nt_map = {}
        if "bodyId" in self.neurotransmitters.columns:
            if "neurotransmitter" in self.neurotransmitters.columns:
                nt_map = dict(
                    zip(
                        self.neurotransmitters["bodyId"],
                        self.neurotransmitters["neurotransmitter"]
                    )
                )
        
        logger.info(f"Mapped {len(nt_map)} neurons to neurotransmitters")
        return nt_map

    def get_statistics(self) -> Dict:
        """Get basic statistics about the connectome."""
        stats = {
            "num_neurons": len(self.neurons) if self.neurons is not None else 0,
            "num_synapses": len(self.synapses) if self.synapses is not None else 0,
        }

        if self.synapses is not None and "weight" in self.synapses.columns:
            stats["weight_stats"] = {
                "mean": float(self.synapses["weight"].mean()),
                "std": float(self.synapses["weight"].std()),
                "min": float(self.synapses["weight"].min()),
                "max": float(self.synapses["weight"].max()),
            }

        if self.neurons is not None and "size" in self.neurons.columns:
            stats["neuron_size_stats"] = {
                "mean": float(self.neurons["size"].mean()),
                "std": float(self.neurons["size"].std()),
            }

        return stats

    def summary(self) -> None:
        """Print data loading summary."""
        logger.info("\n" + "="*60)
        logger.info("CONNECTOME DATA SUMMARY")
        logger.info("="*60)
        
        stats = self.get_statistics()
        logger.info(f"Neurons: {stats['num_neurons']:,}")
        logger.info(f"Synapses: {stats['num_synapses']:,}")
        
        if "weight_stats" in stats:
            ws = stats["weight_stats"]
            logger.info(f"Synapse weights - Mean: {ws['mean']:.2f}, "
                       f"Std: {ws['std']:.2f}, "
                       f"Range: [{ws['min']:.2f}, {ws['max']:.2f}]")
        
        neuron_types = self.get_neuron_types()
        logger.info(f"Neuron types: {len(neuron_types)}")
        for ntype, ids in list(neuron_types.items())[:10]:
            logger.info(f"  {ntype}: {len(ids)} neurons")
        
        logger.info("="*60 + "\n")


if __name__ == "__main__":
    loader = ConnectomeDataLoader()
    logger.info("Janelia male-CNS connectome data loader")
    logger.info("\nUsage:")
    logger.info("  1. Download data from https://male-cns.janelia.org/download/")
    logger.info("  2. Place files in: data/raw/")
    logger.info("  3. Run: loader.load_feather_files()")
