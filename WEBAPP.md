# Drosophila LIF Brain Simulator - Web App

## 🌐 Live Demo

**Access the simulator here:** 
https://fengzhou2520-debug.github.io/drosophila-lif-connectome/

Simply open this URL in your browser - no installation required!

## ✨ Features

The web-based simulator includes:

### Simulation Controls
- **Network Size Selection**: Choose between small (100), medium (500), or large (2000) neuron networks
- **Adjustable Parameters**:
  - Simulation duration (100-5000 ms)
  - Timestep resolution (0.01-1 ms)
  - Input stimulus amplitude (0-500 pA)
  - Stimulus type selection (None, Poisson, Step Current)

### Stimulus Types
1. **No Stimulus**: Spontaneous network activity
2. **Poisson Input**: Random spike trains at 5 Hz to 10% of neurons
3. **Step Current**: Constant current injection during simulation

### Visualizations
- **Spike Raster**: Individual neuron spike times (classic neuroscience plot)
- **Population Activity**: Network spike count over time
- **Firing Rate Distribution**: Histogram of neuron firing rates
- **Interspike Interval (ISI)**: Distribution of time intervals between spikes

### Statistics Panel
Real-time computation and display of:
- Total neurons in network
- Number of active neurons
- Total spike count
- Mean firing rate (Hz)
- Maximum firing rate (Hz)
- Simulation duration

## 🔧 Technical Details

### Architecture

The simulator runs entirely in your browser using:
- **HTML5** for interface structure
- **CSS3** for responsive styling
- **JavaScript (ES6+)** for the LIF simulation engine
- **Plotly.js** for interactive visualizations

### Neural Model

The simulator implements the **Leaky Integrate-and-Fire (LIF)** neuron model:

```
C_m * dV/dt = g_leak * (E_leak - V) + I_syn + I_ext
```

When V > V_threshold:
- Generate action potential (spike)
- Reset V → V_reset
- Enter refractory period (τ_ref = 1.5 ms)

### Synaptic Model

Each synapse features:
- **Exponential decay**: g_syn(t) = g_syn(0) * exp(-t/τ_decay)
- **Neurotransmitter-specific reversal potentials**:
  - Excitatory (Glutamate): E_rev = 0 mV
  - Inhibitory (GABA): E_rev = -70 mV
- **Dynamic conductance**: Changes based on presynaptic spike timing

### Network Connectivity

Random connectivity with:
- ~2 synapses per neuron (biologically plausible)
- Mix of excitatory (60%) and inhibitory (40%) synapses
- Synaptic weights uniformly distributed [0.2, 1.0]

## 📊 How to Use

1. **Select network size** - Start with "Medium" for balanced speed/detail
2. **Set simulation duration** - 500 ms is good for quick tests
3. **Choose stimulus type** - "Poisson" is realistic for sensory input
4. **Adjust amplitude** - 50 pA is a typical starting value
5. **Click "Run"** - Simulation starts (takes 5-30 seconds depending on size)
6. **View results** - Switch between tabs to explore different visualizations

### Example Scenarios

**Scenario 1: Spontaneous Activity**
- Network Size: Medium (500 neurons)
- Duration: 500 ms
- Stimulus: None
- Observe: Natural oscillations and bursting patterns

**Scenario 2: Sensory Response**
- Network Size: Large (2000 neurons)
- Duration: 1000 ms
- Stimulus: Poisson (5 Hz)
- Observe: Increased population firing rates

**Scenario 3: Strong Drive**
- Network Size: Small (100 neurons)
- Duration: 500 ms
- Stimulus: Step Current (200 pA)
- Observe: Synchronized population responses

## ⚡ Performance

| Network Size | Simulation Time | Estimated Duration |
|--------------|-----------------|-------------------|
| Small (100)  | 500 ms          | ~5 seconds        |
| Medium (500) | 500 ms          | ~15 seconds       |
| Large (2000) | 500 ms          | ~30 seconds       |

**Browser Requirements:**
- Modern browser with ES6 support (Chrome, Firefox, Safari, Edge)
- ~50 MB of RAM for large simulations
- Recommended screen resolution: 1280x800 or larger

## 🧮 Mathematical Details

### Euler Integration
The simulator uses forward Euler method with timestep dt:

```
V_new = V_old + (dV/dt) * dt
```

Where:
```
dV/dt = [g_leak * (E_leak - V) + I_syn + I_ext] / C_m / tau_m
```

### Spike Detection
Action potential generated when:
```
V(t) > V_threshold AND t - t_last_spike > tau_refractory
```

### Synaptic Current
```
I_syn = Σ g_syn_i * (V - E_rev_i)
```

On presynaptic spike:
```
g_syn += weight * amplitude
```

## 📈 Interpreting Results

### Spike Raster
- **X-axis**: Time in milliseconds
- **Y-axis**: Neuron index
- **Black dots**: Individual spike times
- **Interpretation**: Dense raster = high network activity; gaps = quiet periods

### Population Activity
- **X-axis**: Time (ms)
- **Y-axis**: Number of spikes in each timestep
- **Interpretation**: Peaks = synchronized firing; baseline = spontaneous activity

### Firing Rate Distribution
- **X-axis**: Firing rate in Hz
- **Y-axis**: Number of neurons with that rate
- **Interpretation**: Wider distribution = more diversity; peaked = homogeneous

### ISI Distribution
- **X-axis**: Time between consecutive spikes (log scale)
- **Y-axis**: Count (log scale)
- **Interpretation**: Regular firing = sharp peak; irregular = broad distribution

## 🔬 Biological Realism

The simulator uses parameters based on:
- Drosophila melanogaster neuron physiology
- Typical insect brain connectivity patterns
- Hodgkin-Huxley-inspired biophysics

**Known Simplifications:**
- Single compartment (no dendritic structure)
- No voltage-dependent ion channels
- Instantaneous synaptic transmission (no rise time)
- Random connectivity (not based on actual connectome)

For full connectome-based simulations with actual Janelia data, see the Python implementation in the `/src` directory.

## 🐛 Troubleshooting

### Simulator runs very slowly
- Reduce network size to "Small"
- Reduce simulation duration
- Ensure no other browser tabs are running heavy computations

### No spikes observed
- Increase stimulus amplitude
- Change stimulus type to "Poisson"
- Verify that network isn't in stuck refractory state (try Reset)

### Visualization not updating
- Check browser console for errors (F12)
- Clear browser cache and reload
- Try a different browser

### Browser crashes with large network
- Use smaller network size
- Reduce duration
- Close other applications to free RAM

## 🔗 Related Resources

- **Main Repository**: https://github.com/fengzhou2520-debug/drosophila-lif-connectome
- **Python Implementation**: See `/src` directory for full-featured simulator
- **Janelia Connectome**: https://male-cns.janelia.org/
- **LIF Theory**: https://en.wikipedia.org/wiki/Leaky_integrate-and-fire
- **Drosophila Brain**: https://virtualflybrain.org/

## 📝 Citation

If you use this simulator in research or teaching, please cite:

```
Feng Zhou, "Drosophila LIF Brain Simulator - Web App", 2026
https://github.com/fengzhou2520-debug/drosophila-lif-connectome
```

## 📄 License

MIT License - Feel free to use, modify, and distribute

---

**Enjoy exploring the fruit fly brain!** 🪰🧠
