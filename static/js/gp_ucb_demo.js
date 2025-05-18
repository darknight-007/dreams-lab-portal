class GPUCBDemo {
    constructor() {
        this.canvas = document.getElementById('gpucbCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // GP-UCB parameters
        this.beta = 2.0;  // Exploration parameter
        this.lengthScale = 1.0;
        this.noiseLevel = 0.01;
        this.testFunction = 'sine';
        this.maxIterations = 20;
        this.currentIteration = 0;
        
        // Data
        this.data = { x: [], y: [], values: [] };
        this.gp = null;
        this.bestPoint = null;
        this.bestValue = -Infinity;
        
        // Animation
        this.isAnimating = false;
        this.animationFrame = null;
        
        // Cache for performance
        this.cachedTestPoints = Array.from({length: 200}, (_, i) => -5 + 10 * i/199);
        this.cachedTrueValues = this.cachedTestPoints.map(x => this.evaluateFunction(x));
        
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        // GP-UCB parameters
        document.getElementById('beta').addEventListener('input', (e) => {
            this.beta = Math.exp(parseFloat(e.target.value));
            e.target.nextElementSibling.textContent = this.beta.toFixed(2);
            this.updateGP();
            this.draw();
        });
        
        document.getElementById('lengthScale').addEventListener('input', (e) => {
            this.lengthScale = Math.exp(parseFloat(e.target.value));
            e.target.nextElementSibling.textContent = this.lengthScale.toFixed(2);
            this.updateGP();
            this.draw();
        });
        
        document.getElementById('noiseLevel').addEventListener('input', (e) => {
            this.noiseLevel = Math.exp(parseFloat(e.target.value));
            e.target.nextElementSibling.textContent = this.noiseLevel.toFixed(3);
            this.updateGP();
            this.draw();
        });
        
        // Test function
        document.querySelectorAll('input[name="testFunction"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.testFunction = e.target.value;
                this.cachedTrueValues = this.cachedTestPoints.map(x => this.evaluateFunction(x));
                this.reset();
                this.draw();
            });
        });
        
        // Controls
        document.getElementById('reset').addEventListener('click', () => {
            this.reset();
            this.draw();
        });
        
        document.getElementById('step').addEventListener('click', () => {
            this.step();
            this.draw();
        });
        
        document.getElementById('optimize').addEventListener('click', () => {
            if(this.isAnimating) {
                this.stopAnimation();
            } else {
                this.startAnimation();
            }
        });
    }
    
    evaluateFunction(x) {
        switch(this.testFunction) {
            case 'sine':
                return Math.sin(2 * Math.PI * x / 4);
            case 'bump':
                return Math.exp(-0.5 * x * x);
            case 'step':
                return x < 0 ? -1 : 1;
            default:
                return Math.sin(2 * Math.PI * x / 4);
        }
    }
    
    reset() {
        this.data = { x: [], y: [], values: [] };
        this.gp = null;
        this.bestPoint = null;
        this.bestValue = -Infinity;
        this.currentIteration = 0;
        document.getElementById('optimize').textContent = 'Start Optimization';
        this.stopAnimation();
    }
    
    step() {
        if(this.currentIteration >= this.maxIterations) {
            this.stopAnimation();
            return;
        }
        
        // First point is random
        if(this.data.x.length === 0) {
            const x = -4 + Math.random() * 8;
            this.addPoint(x);
            return;
        }
        
        // Find next point using GP-UCB
        let bestUCB = -Infinity;
        let bestX = null;
        
        for(let i = 0; i < this.cachedTestPoints.length; i++) {
            const x = this.cachedTestPoints[i];
            const prediction = this.gp.predict(x);
            const ucb = prediction.mean + this.beta * Math.sqrt(prediction.variance);
            
            if(ucb > bestUCB) {
                bestUCB = ucb;
                bestX = x;
            }
        }
        
        this.addPoint(bestX);
    }
    
    addPoint(x) {
        const y = this.evaluateFunction(x) + Math.sqrt(this.noiseLevel) * (Math.random() * 2 - 1);
        
        this.data.x.push(x);
        this.data.y.push(y);
        this.data.values.push(this.evaluateFunction(x));
        
        if(this.data.values[this.data.values.length - 1] > this.bestValue) {
            this.bestValue = this.data.values[this.data.values.length - 1];
            this.bestPoint = x;
        }
        
        this.currentIteration++;
        this.updateGP();
        this.updateMetrics();
    }
    
    updateGP() {
        if(this.data.x.length === 0) return;
        
        const kernel = (x1, x2) => 
            Math.exp(-0.5 * Math.pow(x1 - x2, 2) / (this.lengthScale * this.lengthScale));
        
        this.gp = new GaussianProcess(kernel, this.noiseLevel);
        this.gp.fit(this.data.x, this.data.y);
    }
    
    updateMetrics() {
        // Update display
        document.getElementById('iterationValue').textContent = this.currentIteration;
        document.getElementById('bestValue').textContent = this.bestValue.toFixed(3);
        document.getElementById('bestPoint').textContent = this.bestPoint.toFixed(3);
    }
    
    draw() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        this.drawGrid();
        
        // Draw true function
        this.drawTrueFunction();
        
        if(this.gp) {
            // Draw uncertainty region
            this.drawUncertaintyRegion();
            
            // Draw GP prediction
            this.drawPrediction();
            
            // Draw acquisition function
            this.drawAcquisitionFunction();
        }
        
        // Draw data points
        this.drawDataPoints();
        
        // Draw best point
        if(this.bestPoint !== null) {
            this.drawBestPoint();
        }
    }
    
    drawGrid() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = '#ddd';
        this.ctx.lineWidth = 1;
        
        // Draw grid lines
        for(let x = 0; x <= 10; x++) {
            const xPos = 50 + (width - 100) * x / 10;
            this.ctx.beginPath();
            this.ctx.moveTo(xPos, 50);
            this.ctx.lineTo(xPos, height - 50);
            this.ctx.stroke();
        }
        
        for(let y = 0; y <= 10; y++) {
            const yPos = 50 + (height - 100) * y / 10;
            this.ctx.beginPath();
            this.ctx.moveTo(50, yPos);
            this.ctx.lineTo(width - 50, yPos);
            this.ctx.stroke();
        }
        
        // Draw axes
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 2;
        
        // X axis
        this.ctx.beginPath();
        this.ctx.moveTo(50, height/2);
        this.ctx.lineTo(width - 50, height/2);
        this.ctx.stroke();
        
        // Y axis
        this.ctx.beginPath();
        this.ctx.moveTo(width/2, 50);
        this.ctx.lineTo(width/2, height - 50);
        this.ctx.stroke();
        
        // Labels
        this.ctx.fillStyle = '#000';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        this.ctx.font = '12px Arial';
        
        // X axis labels
        for(let x = -5; x <= 5; x++) {
            const xPos = width/2 + (width - 100) * x / 10;
            this.ctx.fillText(x.toString(), xPos, height/2 + 5);
        }
        
        // Y axis labels
        this.ctx.textAlign = 'right';
        this.ctx.textBaseline = 'middle';
        for(let y = -5; y <= 5; y++) {
            const yPos = height/2 - (height - 100) * y / 10;
            this.ctx.fillText(y.toString(), width/2 - 5, yPos);
        }
    }
    
    drawTrueFunction() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = '#666';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        
        for(let i = 0; i < this.cachedTestPoints.length; i++) {
            const x = 50 + (width - 100) * (i / (this.cachedTestPoints.length - 1));
            const y = height/2 - (height - 100) * this.cachedTrueValues[i] / 10;
            
            if(i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }
    
    drawUncertaintyRegion() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.fillStyle = 'rgba(0,123,255,0.2)';
        this.ctx.beginPath();
        
        // Upper bound
        let x = 50;
        for(let i = 0; i < this.cachedTestPoints.length; i++) {
            const prediction = this.gp.predict(this.cachedTestPoints[i]);
            const y = height/2 - (height - 100) * (prediction.mean + 2 * Math.sqrt(prediction.variance)) / 10;
            
            if(i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
            
            x += (width - 100) / (this.cachedTestPoints.length - 1);
        }
        
        // Lower bound
        x = width - 50;
        for(let i = this.cachedTestPoints.length - 1; i >= 0; i--) {
            const prediction = this.gp.predict(this.cachedTestPoints[i]);
            const y = height/2 - (height - 100) * (prediction.mean - 2 * Math.sqrt(prediction.variance)) / 10;
            
            this.ctx.lineTo(x, y);
            x -= (width - 100) / (this.cachedTestPoints.length - 1);
        }
        
        this.ctx.closePath();
        this.ctx.fill();
    }
    
    drawPrediction() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = '#007bff';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        let x = 50;
        for(let i = 0; i < this.cachedTestPoints.length; i++) {
            const prediction = this.gp.predict(this.cachedTestPoints[i]);
            const y = height/2 - (height - 100) * prediction.mean / 10;
            
            if(i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
            
            x += (width - 100) / (this.cachedTestPoints.length - 1);
        }
        
        this.ctx.stroke();
    }
    
    drawAcquisitionFunction() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = '#28a745';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        
        let x = 50;
        for(let i = 0; i < this.cachedTestPoints.length; i++) {
            const prediction = this.gp.predict(this.cachedTestPoints[i]);
            const ucb = prediction.mean + this.beta * Math.sqrt(prediction.variance);
            const y = height/2 - (height - 100) * ucb / 10;
            
            if(i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
            
            x += (width - 100) / (this.cachedTestPoints.length - 1);
        }
        
        this.ctx.stroke();
    }
    
    drawDataPoints() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.fillStyle = '#000';
        for(let i = 0; i < this.data.x.length; i++) {
            const x = width/2 + (width - 100) * this.data.x[i] / 10;
            const y = height/2 - (height - 100) * this.data.y[i] / 10;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 4, 0, 2*Math.PI);
            this.ctx.fill();
        }
    }
    
    drawBestPoint() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        const x = width/2 + (width - 100) * this.bestPoint / 10;
        const y = height/2 - (height - 100) * this.bestValue / 10;
        
        this.ctx.strokeStyle = '#dc3545';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(x, y, 6, 0, 2*Math.PI);
        this.ctx.stroke();
    }
    
    startAnimation() {
        this.isAnimating = true;
        document.getElementById('optimize').textContent = 'Stop Optimization';
        this.animate();
    }
    
    stopAnimation() {
        this.isAnimating = false;
        document.getElementById('optimize').textContent = 'Start Optimization';
        if(this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    animate() {
        if(!this.isAnimating) return;
        
        this.step();
        this.draw();
        
        if(this.currentIteration < this.maxIterations) {
            this.animationFrame = requestAnimationFrame(() => this.animate());
        } else {
            this.stopAnimation();
        }
    }
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const demo = new GPUCBDemo();
}); 