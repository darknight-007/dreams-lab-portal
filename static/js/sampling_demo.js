class SamplingDemo {
    constructor() {
        this.canvas = document.getElementById('sampling-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size
        this.canvas.width = 600;
        this.canvas.height = 600;
        
        // Optimization parameters
        this.populationSize = 50;
        this.eliteFraction = 0.2;
        this.noiseLevel = 0.1;
        this.contourLevels = 15;
        
        // Search space bounds
        this.bounds = {
            x: { min: -5, max: 5 },
            y: { min: -5, max: 5 }
        };
        
        // Distribution parameters
        this.distribution = {
            mean: { x: 0, y: 0 },
            std: { x: 2, y: 2 }
        };
        
        // Optimization state
        this.samples = [];
        this.bestSample = null;
        this.iteration = 0;
        this.isRunning = false;
        
        // Setup
        this.setupUI();
        this.reset();
        this.draw();
    }
    
    setupUI() {
        // Button event listeners
        document.getElementById('reset-btn').addEventListener('click', () => this.reset());
        document.getElementById('step-btn').addEventListener('click', () => this.step());
        document.getElementById('run-btn').addEventListener('click', () => this.run());
        document.getElementById('pause-btn').addEventListener('click', () => this.pause());
        
        // Slider event listeners
        document.getElementById('population-size').addEventListener('input', (e) => {
            this.populationSize = parseInt(e.target.value);
            document.getElementById('population-size-value').textContent = this.populationSize;
        });
        
        document.getElementById('elite-fraction').addEventListener('input', (e) => {
            this.eliteFraction = parseInt(e.target.value) / 100;
            document.getElementById('elite-fraction-value').textContent = this.eliteFraction.toFixed(2);
        });
        
        document.getElementById('noise-level').addEventListener('input', (e) => {
            this.noiseLevel = parseInt(e.target.value) / 100;
            document.getElementById('noise-level-value').textContent = this.noiseLevel.toFixed(2);
        });
        
        document.getElementById('contour-levels').addEventListener('input', (e) => {
            this.contourLevels = parseInt(e.target.value);
            document.getElementById('contour-levels-value').textContent = this.contourLevels;
            this.draw();
        });
    }
    
    objectiveFunction(x, y) {
        // Example objective function: negative quadratic + sinusoidal
        return -((x-2)*(x-2) + (y-3)*(y-3)) + 5 * Math.sin(x) * Math.cos(y);
    }
    
    reset() {
        // Reset distribution to initial state
        this.distribution = {
            mean: { x: 0, y: 0 },
            std: { x: 2, y: 2 }
        };
        
        // Reset optimization state
        this.samples = [];
        this.bestSample = null;
        this.iteration = 0;
        this.isRunning = false;
        
        // Update UI
        document.getElementById('iteration-count').textContent = '0';
        document.getElementById('best-value').textContent = '-';
        document.getElementById('mean-value').textContent = '-';
        document.getElementById('distribution-mean').textContent = '(0, 0)';
        document.getElementById('distribution-std').textContent = '(2, 2)';
        
        // Enable/disable buttons
        document.getElementById('run-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;
        
        this.draw();
    }
    
    generateSamples() {
        this.samples = [];
        for (let i = 0; i < this.populationSize; i++) {
            // Generate sample from current distribution
            const x = this.distribution.mean.x + this.distribution.std.x * this.randn();
            const y = this.distribution.mean.y + this.distribution.std.y * this.randn();
            
            // Evaluate objective function with noise
            const value = this.objectiveFunction(x, y) + this.noiseLevel * this.randn();
            
            this.samples.push({ x, y, value });
        }
        
        // Sort samples by value
        this.samples.sort((a, b) => b.value - a.value);
        
        // Update best sample if needed
        if (!this.bestSample || this.samples[0].value > this.bestSample.value) {
            this.bestSample = { ...this.samples[0] };
        }
    }
    
    updateDistribution() {
        // Select elite samples
        const numElite = Math.max(2, Math.floor(this.populationSize * this.eliteFraction));
        const eliteSamples = this.samples.slice(0, numElite);
        
        // Compute new mean
        const newMean = {
            x: eliteSamples.reduce((sum, s) => sum + s.x, 0) / numElite,
            y: eliteSamples.reduce((sum, s) => sum + s.y, 0) / numElite
        };
        
        // Compute new standard deviation
        const newStd = {
            x: Math.sqrt(eliteSamples.reduce((sum, s) => sum + (s.x - newMean.x) ** 2, 0) / numElite),
            y: Math.sqrt(eliteSamples.reduce((sum, s) => sum + (s.y - newMean.y) ** 2, 0) / numElite)
        };
        
        // Update distribution with some minimum std to prevent collapse
        this.distribution.mean = newMean;
        this.distribution.std = {
            x: Math.max(0.1, newStd.x),
            y: Math.max(0.1, newStd.y)
        };
    }
    
    step() {
        this.generateSamples();
        this.updateDistribution();
        this.iteration++;
        
        // Update UI
        document.getElementById('iteration-count').textContent = this.iteration;
        document.getElementById('best-value').textContent = this.bestSample.value.toFixed(4);
        document.getElementById('mean-value').textContent = this.samples[0].value.toFixed(4);
        document.getElementById('distribution-mean').textContent = 
            `(${this.distribution.mean.x.toFixed(2)}, ${this.distribution.mean.y.toFixed(2)})`;
        document.getElementById('distribution-std').textContent = 
            `(${this.distribution.std.x.toFixed(2)}, ${this.distribution.std.y.toFixed(2)})`;
        
        this.draw();
    }
    
    async run() {
        this.isRunning = true;
        document.getElementById('run-btn').disabled = true;
        document.getElementById('pause-btn').disabled = false;
        
        while (this.isRunning) {
            this.step();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
    
    pause() {
        this.isRunning = false;
        document.getElementById('run-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw contours of objective function
        this.drawContours();
        
        // Draw samples
        this.samples.forEach(sample => {
            this.drawPoint(sample.x, sample.y, 'blue', 3);
        });
        
        // Draw best sample
        if (this.bestSample) {
            this.drawPoint(this.bestSample.x, this.bestSample.y, 'red', 5);
        }
        
        // Draw distribution
        this.drawDistribution();
    }
    
    drawContours() {
        const resolution = 50;
        const dx = (this.bounds.x.max - this.bounds.x.min) / resolution;
        const dy = (this.bounds.y.max - this.bounds.y.min) / resolution;
        
        // Compute function values
        const values = [];
        for (let i = 0; i <= resolution; i++) {
            values[i] = [];
            for (let j = 0; j <= resolution; j++) {
                const x = this.bounds.x.min + i * dx;
                const y = this.bounds.y.min + j * dy;
                values[i][j] = this.objectiveFunction(x, y);
            }
        }
        
        // Find min and max values
        let min = Infinity, max = -Infinity;
        values.forEach(row => {
            row.forEach(val => {
                min = Math.min(min, val);
                max = Math.max(max, val);
            });
        });
        
        // Draw contours
        const levels = [];
        for (let i = 0; i < this.contourLevels; i++) {
            levels.push(min + (max - min) * i / (this.contourLevels - 1));
        }
        
        levels.forEach((level, index) => {
            const alpha = 0.1 + 0.2 * index / (this.contourLevels - 1);
            this.ctx.strokeStyle = `rgba(0, 0, 0, ${alpha})`;
            this.ctx.beginPath();
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const x1 = this.worldToCanvas(this.bounds.x.min + i * dx, 'x');
                    const x2 = this.worldToCanvas(this.bounds.x.min + (i + 1) * dx, 'x');
                    const y1 = this.worldToCanvas(this.bounds.y.min + j * dy, 'y');
                    const y2 = this.worldToCanvas(this.bounds.y.min + (j + 1) * dy, 'y');
                    
                    const v11 = values[i][j];
                    const v12 = values[i][j + 1];
                    const v21 = values[i + 1][j];
                    const v22 = values[i + 1][j + 1];
                    
                    if ((v11 <= level && level < v12) || (v12 <= level && level < v11)) {
                        const t = (level - v11) / (v12 - v11);
                        const y = y1 + t * (y2 - y1);
                        this.ctx.moveTo(x1, y);
                        this.ctx.lineTo(x1, y);
                    }
                    if ((v11 <= level && level < v21) || (v21 <= level && level < v11)) {
                        const t = (level - v11) / (v21 - v11);
                        const x = x1 + t * (x2 - x1);
                        this.ctx.moveTo(x, y1);
                        this.ctx.lineTo(x, y1);
                    }
                }
            }
            
            this.ctx.stroke();
        });
    }
    
    drawPoint(x, y, color, size) {
        const cx = this.worldToCanvas(x, 'x');
        const cy = this.worldToCanvas(y, 'y');
        
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, size, 0, 2 * Math.PI);
        this.ctx.fill();
    }
    
    drawDistribution() {
        // Draw ellipse representing the distribution
        const cx = this.worldToCanvas(this.distribution.mean.x, 'x');
        const cy = this.worldToCanvas(this.distribution.mean.y, 'y');
        const rx = Math.abs(this.worldToCanvas(this.distribution.std.x, 'x') - this.worldToCanvas(0, 'x'));
        const ry = Math.abs(this.worldToCanvas(this.distribution.std.y, 'y') - this.worldToCanvas(0, 'y'));
        
        this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
        this.ctx.beginPath();
        this.ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
        this.ctx.stroke();
    }
    
    worldToCanvas(value, axis) {
        const bounds = this.bounds[axis];
        const size = axis === 'x' ? this.canvas.width : this.canvas.height;
        return size * (value - bounds.min) / (bounds.max - bounds.min);
    }
    
    randn() {
        // Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

// Initialize demo when the page loads
window.addEventListener('load', () => {
    const demo = new SamplingDemo();
}); 