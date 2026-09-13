/**
 * Drosophila LIF Brain Simulator - JavaScript Engine
 * Implements leaky integrate-and-fire neural network in the browser
 */

let simulation = {
    running: false,
    data: null,
    neurons: [],
    synapses: [],
    time: 0,
    history: {
        time: [],
        voltages: [],
        spikes: [],
        activity: []
    }
};

// LIF Neuron Parameters
const neuronParams = {
    C_m: 1.0,
    g_leak: 0.1,
    E_leak: -65.0,
    V_th: -50.0,
    V_reset: -65.0,
    tau_m: 15.0,
    tau_ref: 1.5,
    area: 1000.0
};

// Neuron class
class LIFNeuron {
    constructor(id) {
        this.id = id;
        this.V = neuronParams.E_leak;
        this.I_syn = 0;
        this.I_ext = 0;
        this.last_spike = -Infinity;
        this.spike_times = [];
    }
    
    step(dt, time) {
        const in_refractory = (time - this.last_spike) < neuronParams.tau_ref;
        
        if (in_refractory) {
            this.V = neuronParams.V_reset;
            return false;
        }
        
        const I_total = (this.I_syn + this.I_ext) / 1000.0;
        const R = 1.0 / neuronParams.g_leak;
        const dV = (neuronParams.E_leak - this.V + R * I_total) / neuronParams.tau_m;
        
        this.V += dV * dt;
        this.V = Math.max(-150, Math.min(100, this.V));
        
        let spiked = false;
        if (this.V > neuronParams.V_th) {
            spiked = true;
            this.spike_times.push(time);
            this.last_spike = time;
            this.V = neuronParams.V_reset;
        }
        
        return spiked;
    }
    
    reset() {
        this.V = neuronParams.E_leak;
        this.I_syn = 0;
        this.I_ext = 0;
        this.last_spike = -Infinity;
        this.spike_times = [];
    }
}

// Synapse class
class Synapse {
    constructor(pre_id, post_id, weight = 1.0) {
        this.pre_id = pre_id;
        this.post_id = post_id;
        this.weight = weight;
        this.g_syn = 0;
        this.E_rev = pre_id % 3 === 0 ? -70 : 0; // GABA vs Glutamate
        this.tau_decay = 5.0;
    }
    
    step(dt, post_V) {
        this.g_syn *= Math.exp(-dt / this.tau_decay);
        const I = this.g_syn * (post_V - this.E_rev);
        return I;
    }
    
    deliver_spike(amplitude = 2.0) {
        this.g_syn += amplitude * this.weight;
    }
}

// Initialize network
function initializeNetwork(size) {
    simulation.neurons = [];
    simulation.synapses = [];
    
    const neuronCount = size === 'small' ? 100 : size === 'medium' ? 500 : 2000;
    
    // Create neurons
    for (let i = 0; i < neuronCount; i++) {
        simulation.neurons.push(new LIFNeuron(i));
    }
    
    // Create random connectivity
    const synapseCount = Math.floor(neuronCount * 2); // ~2 synapses per neuron
    for (let i = 0; i < synapseCount; i++) {
        const pre = Math.floor(Math.random() * neuronCount);
        const post = Math.floor(Math.random() * neuronCount);
        if (pre !== post) {
            simulation.synapses.push(new Synapse(pre, post, Math.random() * 0.8 + 0.2));
        }
    }
    
    console.log(`Network initialized: ${neuronCount} neurons, ${simulation.synapses.length} synapses`);
}

// Reset simulation
function resetSimulation() {
    simulation.running = false;
    simulation.time = 0;
    simulation.neurons.forEach(n => n.reset());
    simulation.history = {
        time: [],
        voltages: [],
        spikes: [],
        activity: []
    };
    document.getElementById('status').innerHTML = '';
    document.getElementById('runButton').disabled = false;
    document.getElementById('runButton').textContent = '▶ Run';
}

// Generate stimulus
function getStimulus(time, neuronId, stimulusType, amplitude) {
    if (stimulusType === 'none') return 0;
    
    if (stimulusType === 'poisson') {
        // Poisson stimulus to 10% of neurons
        if (neuronId % 10 === 0) {
            return Math.random() < 0.01 ? amplitude : 0;
        }
    }
    
    if (stimulusType === 'step') {
        if (time > 100 && time < 400) {
            return neuronId % 5 === 0 ? amplitude : 0;
        }
    }
    
    return 0;
}

// Run simulation
async function runSimulation() {
    const networkSize = document.getElementById('networkSize').value;
    const duration = parseFloat(document.getElementById('duration').value);
    const dt = parseFloat(document.getElementById('dt').value);
    const stimulusType = document.getElementById('stimulusType').value;
    const amplitude = parseFloat(document.getElementById('amplitude').value);
    
    document.getElementById('runButton').disabled = true;
    document.getElementById('runButton').textContent = '⏳ Running...';
    
    resetSimulation();
    initializeNetwork(networkSize);
    
    updateStatus('Running simulation...', 'info');
    
    // Run simulation
    const nSteps = Math.floor(duration / dt);
    simulation.history.time = [];
    simulation.history.voltages = [];
    simulation.history.spikes = [];
    simulation.history.activity = [];
    
    for (let step = 0; step < nSteps; step++) {
        simulation.time = step * dt;
        
        // Apply stimulus
        simulation.neurons.forEach((neuron, i) => {
            neuron.I_ext = getStimulus(simulation.time, i, stimulusType, amplitude);
        });
        
        // Compute synaptic currents
        const I_syn = new Array(simulation.neurons.length).fill(0);
        simulation.synapses.forEach(syn => {
            const I = syn.step(dt, simulation.neurons[syn.post_id].V);
            I_syn[syn.post_id] += I;
        });
        
        simulation.neurons.forEach((neuron, i) => {
            neuron.I_syn = I_syn[i];
        });
        
        // Neural integration
        let spike_count = 0;
        simulation.neurons.forEach(neuron => {
            const spiked = neuron.step(dt, simulation.time);
            if (spiked) {
                spike_count++;
                // Deliver spike to postsynaptic neurons
                simulation.synapses.forEach(syn => {
                    if (syn.pre_id === neuron.id) {
                        syn.deliver_spike();
                    }
                });
            }
        });
        
        // Record data
        simulation.history.time.push(simulation.time);
        simulation.history.activity.push(spike_count);
        
        // Sample voltages every 10 steps
        if (step % 10 === 0) {
            const voltages = simulation.neurons.map(n => n.V).slice(0, Math.min(50, simulation.neurons.length));
            simulation.history.voltages.push(voltages);
        }
        
        // Record spikes
        const spikes = [];
        simulation.neurons.forEach((neuron, i) => {
            neuron.spike_times.forEach(t => {
                if (t === simulation.time) {
                    spikes.push([i, t]);
                }
            });
        });
        if (spikes.length > 0) {
            simulation.history.spikes.push(...spikes);
        }
        
        // Update progress every 5%
        if (step % Math.max(1, Math.floor(nSteps / 20)) === 0) {
            const progress = Math.floor((step / nSteps) * 100);
            updateStatus(`Simulating... ${progress}%`, 'info');
            await new Promise(resolve => setTimeout(resolve, 0)); // Yield
        }
    }
    
    updateStatus('Simulation complete!', 'success');
    document.getElementById('runButton').disabled = false;
    document.getElementById('runButton').textContent = '▶ Run';
    
    // Generate plots
    generatePlots();
}

// Generate statistics
function generateStats() {
    if (simulation.neurons.length === 0) return {};
    
    const all_spikes = [];
    simulation.neurons.forEach(n => all_spikes.push(...n.spike_times));
    
    let firing_rates = [];
    const duration_sec = simulation.time / 1000;
    simulation.neurons.forEach(n => {
        if (duration_sec > 0) {
            firing_rates.push(n.spike_times.length / duration_sec);
        }
    });
    
    const active_neurons = simulation.neurons.filter(n => n.spike_times.length > 0).length;
    
    return {
        total_neurons: simulation.neurons.length,
        active_neurons: active_neurons,
        total_spikes: all_spikes.length,
        mean_firing_rate: firing_rates.length > 0 ? (firing_rates.reduce((a,b) => a+b) / firing_rates.length).toFixed(2) : 0,
        max_firing_rate: firing_rates.length > 0 ? Math.max(...firing_rates).toFixed(2) : 0,
        duration: simulation.time.toFixed(1)
    };
}

// Update statistics display
function updateStats() {
    const stats = generateStats();
    const statsHtml = `
        <div class="stat-box">
            <div class="label">Total Neurons</div>
            <div class="value">${stats.total_neurons || 0}</div>
        </div>
        <div class="stat-box">
            <div class="label">Active Neurons</div>
            <div class="value">${stats.active_neurons || 0}</div>
        </div>
        <div class="stat-box">
            <div class="label">Total Spikes</div>
            <div class="value">${stats.total_spikes || 0}</div>
        </div>
        <div class="stat-box">
            <div class="label">Mean Fire Rate</div>
            <div class="value">${stats.mean_firing_rate || 0} Hz</div>
        </div>
        <div class="stat-box">
            <div class="label">Max Fire Rate</div>
            <div class="value">${stats.max_firing_rate || 0} Hz</div>
        </div>
        <div class="stat-box">
            <div class="label">Duration</div>
            <div class="value">${stats.duration || 0} ms</div>
        </div>
    `;
    document.getElementById('statsPanel').innerHTML = statsHtml;
}

// Generate plots
function generatePlots() {
    updateStats();
    
    // Spike Raster
    const spikes_x = [];
    const spikes_y = [];
    simulation.history.spikes.forEach(spike => {
        spikes_x.push(spike[1]);
        spikes_y.push(spike[0]);
    });
    
    const raster_trace = {
        x: spikes_x,
        y: spikes_y,
        mode: 'markers',
        marker: { size: 3, color: 'black' },
        name: 'Spikes'
    };
    
    Plotly.newPlot('rasterPlot', [raster_trace], {
        title: 'Spike Raster Diagram',
        xaxis: { title: 'Time (ms)' },
        yaxis: { title: 'Neuron Index' },
        height: 400,
        margin: { l: 50, r: 20, t: 40, b: 50 }
    }, { responsive: true });
    
    // Population Activity
    const activity_trace = {
        x: simulation.history.time,
        y: simulation.history.activity,
        fill: 'tozeroy',
        name: 'Spike Count'
    };
    
    Plotly.newPlot('activityPlot', [activity_trace], {
        title: 'Population Activity',
        xaxis: { title: 'Time (ms)' },
        yaxis: { title: 'Spikes per timestep' },
        height: 400,
        margin: { l: 50, r: 20, t: 40, b: 50 }
    }, { responsive: true });
    
    // Firing Rate Distribution
    const firing_rates = [];
    const duration_sec = simulation.time / 1000;
    if (duration_sec > 0) {
        simulation.neurons.forEach(n => {
            firing_rates.push(n.spike_times.length / duration_sec);
        });
    }
    
    const fr_trace = {
        x: firing_rates,
        type: 'histogram',
        nbinsx: 30,
        name: 'Firing Rate'
    };
    
    Plotly.newPlot('firingRatePlot', [fr_trace], {
        title: 'Firing Rate Distribution',
        xaxis: { title: 'Firing Rate (Hz)' },
        yaxis: { title: 'Count' },
        height: 350,
        margin: { l: 50, r: 20, t: 40, b: 50 }
    }, { responsive: true });
    
    // ISI Distribution
    const isis = [];
    simulation.neurons.forEach(n => {
        for (let i = 1; i < n.spike_times.length; i++) {
            isis.push(n.spike_times[i] - n.spike_times[i-1]);
        }
    });
    
    const isi_trace = {
        x: isis,
        type: 'histogram',
        nbinsx: 50,
        name: 'ISI'
    };
    
    Plotly.newPlot('isiPlot', [isi_trace], {
        title: 'Interspike Interval Distribution',
        xaxis: { title: 'ISI (ms)', type: 'log' },
        yaxis: { title: 'Count', type: 'log' },
        height: 350,
        margin: { l: 50, r: 20, t: 40, b: 50 }
    }, { responsive: true });
    
    // Summary plot
    const summary_trace = {
        x: simulation.history.time,
        y: simulation.history.activity,
        fill: 'tozeroy',
        name: 'Activity'
    };
    
    Plotly.newPlot('overviewPlot', [summary_trace], {
        title: 'Simulation Summary',
        xaxis: { title: 'Time (ms)' },
        yaxis: { title: 'Spikes per timestep' },
        height: 300,
        margin: { l: 50, r: 20, t: 40, b: 50 }
    }, { responsive: true });
}

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Status message
function updateStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
}
