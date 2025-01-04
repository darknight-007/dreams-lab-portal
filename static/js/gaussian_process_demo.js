class GPDemo {
    constructor() {
        this.canvas = document.getElementById('functionCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.data = { x: [], y: [] };
        
        // Model parameters
        this.kernelType = 'rbf';
        this.lengthScale = 1.0;
        this.signalVariance = 1.0;
        this.noiseLevel = 0.01;
        this.numSamples = 5;
        
        // Cache for kernel computations
        this.cachedK = null;
        this.cachedL = null;
        this.cachedAlpha = null;
        
        // True function parameters
        this.trueFunctionType = null;
        this.trueFunctionParams = null;
        
        this.setupEventListeners();
        this.draw();
    }
    
    setupEventListeners() {
        // Kernel parameters
        document.getElementById('kernelType').addEventListener('change', (e) => {
            this.kernelType = e.target.value;
            this.updateModel();
            this.draw();
        });
        
        document.getElementById('lengthScale').addEventListener('input', (e) => {
            this.lengthScale = Math.exp(parseFloat(e.target.value));
            e.target.nextElementSibling.textContent = this.lengthScale.toFixed(2);
            this.updateModel();
            this.draw();
        });
        
        document.getElementById('signalVariance').addEventListener('input', (e) => {
            this.signalVariance = Math.exp(parseFloat(e.target.value));
            e.target.nextElementSibling.textContent = this.signalVariance.toFixed(2);
            this.updateModel();
            this.draw();
        });
        
        document.getElementById('noiseLevel').addEventListener('input', (e) => {
            this.noiseLevel = Math.exp(parseFloat(e.target.value));
            e.target.nextElementSibling.textContent = this.noiseLevel.toFixed(3);
            this.updateModel();
            this.draw();
        });
        
        document.getElementById('numSamples').addEventListener('input', (e) => {
            this.numSamples = parseInt(e.target.value);
            e.target.nextElementSibling.textContent = this.numSamples;
            this.draw();
        });
        
        // Data controls
        document.getElementById('clearData').addEventListener('click', () => {
            this.data = { x: [], y: [] };
            this.updateModel();
            this.draw();
        });
        
        document.getElementById('generateData').addEventListener('click', () => {
            this.generateRandomData();
            this.updateModel();
            this.draw();
        });
        
        // Example buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => this.loadExample(btn.dataset.example));
        });
        
        // Canvas click handler
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
    }
    
    getKernel() {
        switch(this.kernelType) {
            case 'rbf':
                return (x1, x2) => this.signalVariance * Math.exp(-0.5 * Math.pow(x1 - x2, 2) / (this.lengthScale * this.lengthScale));
            case 'periodic':
                return (x1, x2) => {
                    const period = 4.0;  // Fixed period for demonstration
                    const sinTerm = Math.sin(Math.PI * Math.abs(x1 - x2) / period);
                    return this.signalVariance * Math.exp(-2 * sinTerm * sinTerm / (this.lengthScale * this.lengthScale));
                };
            case 'matern':
                return (x1, x2) => {
                    const d = Math.abs(x1 - x2);
                    const scaled_d = Math.sqrt(3) * d / this.lengthScale;
                    return this.signalVariance * (1 + scaled_d) * Math.exp(-scaled_d);
                };
        }
    }
    
    computeKernelMatrix(X1, X2 = null) {
        const kernel = this.getKernel();
        const n1 = X1.length;
        const n2 = X2 ? X2.length : n1;
        const K = Array(n1).fill().map(() => Array(n2).fill(0));
        
        for(let i = 0; i < n1; i++) {
            for(let j = 0; j < n2; j++) {
                K[i][j] = kernel(X1[i], X2 ? X2[j] : X1[j]);
            }
        }
        
        return K;
    }
    
    updateModel() {
        if(this.data.x.length === 0) {
            this.cachedK = null;
            this.cachedL = null;
            this.cachedAlpha = null;
            return;
        }
        
        // Compute kernel matrix
        this.cachedK = this.computeKernelMatrix(this.data.x);
        
        // Add noise to diagonal
        const K = this.cachedK.map(row => [...row]);
        for(let i = 0; i < K.length; i++) {
            K[i][i] += this.noiseLevel;
        }
        
        // Compute Cholesky decomposition
        this.cachedL = math.lup(K).L;
        
        // Compute alpha = K^(-1)y
        const alpha = math.lusolve(this.cachedL, this.data.y);
        this.cachedAlpha = math.lusolve(math.transpose(this.cachedL), alpha);
    }
    
    predict(xTest) {
        if(this.data.x.length === 0) {
            const k = this.getKernel()(xTest, xTest);
            return {
                mean: 0,
                variance: k + this.noiseLevel
            };
        }
        
        const Kstar = this.computeKernelMatrix([xTest], this.data.x)[0];
        const kss = this.getKernel()(xTest, xTest);
        
        // Compute mean and variance using cached values
        const mean = math.dot(Kstar, this.cachedAlpha);
        const v = math.lusolve(this.cachedL, Kstar);
        const variance = kss - math.dot(v, v) + this.noiseLevel;
        
        return { mean, variance };
    }
    
    draw() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        this.drawGrid();
        
        // Generate test points
        const xTest = Array.from({length: 200}, (_, i) => -5 + 10 * i/199);
        
        // Draw uncertainty region first (background)
        this.drawUncertaintyRegion(xTest);
        
        // Draw mean function
        this.drawMeanFunction(xTest);
        
        // Draw true function if available (on top of mean and uncertainty)
        if (this.trueFunctionType) {
            this.drawTrueFunction(xTest);
        }
        
        // Draw prior samples if no data
        if(this.data.x.length === 0) {
            this.drawPriorSamples(xTest);
        }
        
        // Draw data points (always on top)
        this.drawDataPoints();
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
    
    drawPriorSamples(xTest) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Generate samples from prior
        const K = this.computeKernelMatrix(xTest);
        
        // Add small diagonal term for numerical stability
        for(let i = 0; i < K.length; i++) {
            K[i][i] += 1e-10;
        }
        
        const L = math.lup(K).L;
        
        for(let i = 0; i < this.numSamples; i++) {
            // Generate standard normal samples
            const z = Array(xTest.length).fill(0).map(() => 
                math.random() * Math.sqrt(-2 * Math.log(math.random())) * 
                Math.cos(2 * Math.PI * math.random())
            );
            
            // Transform to multivariate normal
            const f = math.multiply(L, z);
            
            // Draw sample
            this.ctx.strokeStyle = `rgba(0,0,255,0.3)`;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            
            for(let j = 0; j < xTest.length; j++) {
                const x = 50 + (width - 100) * j / (xTest.length - 1);
                const y = height/2 - (height - 100) * f[j] / 10;
                
                if(j === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            
            this.ctx.stroke();
        }
    }
    
    drawUncertaintyRegion(xTest) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.fillStyle = 'rgba(0,123,255,0.2)';
        this.ctx.beginPath();
        
        // Upper bound
        let x = 50;
        for(let i = 0; i < xTest.length; i++) {
            const prediction = this.predict(xTest[i]);
            const y = height/2 - (height - 100) * (prediction.mean + 2 * Math.sqrt(prediction.variance)) / 10;
            
            if(i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
            
            x += (width - 100) / (xTest.length - 1);
        }
        
        // Lower bound
        x = width - 50;
        for(let i = xTest.length - 1; i >= 0; i--) {
            const prediction = this.predict(xTest[i]);
            const y = height/2 - (height - 100) * (prediction.mean - 2 * Math.sqrt(prediction.variance)) / 10;
            
            this.ctx.lineTo(x, y);
            x -= (width - 100) / (xTest.length - 1);
        }
        
        this.ctx.closePath();
        this.ctx.fill();
    }
    
    drawMeanFunction(xTest) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = '#007bff';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        let x = 50;
        for(let i = 0; i < xTest.length; i++) {
            const prediction = this.predict(xTest[i]);
            const y = height/2 - (height - 100) * prediction.mean / 10;
            
            if(i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
            
            x += (width - 100) / (xTest.length - 1);
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
    
    generateRandomData() {
        this.data = { x: [], y: [] };
        const numPoints = 5 + Math.floor(Math.random() * 5);
        
        // Clear true function since we're generating random data
        this.trueFunctionType = null;
        this.trueFunctionParams = null;
        
        for(let i = 0; i < numPoints; i++) {
            const x = -4 + Math.random() * 8;
            let y;
            
            switch(this.kernelType) {
                case 'periodic':
                    y = Math.sin(2 * Math.PI * x / 4) + 0.2 * (Math.random() - 0.5);
                    break;
                case 'rbf':
                default:
                    y = Math.exp(-0.5 * x * x) + 0.2 * (Math.random() - 0.5);
            }
            
            this.data.x.push(x);
            this.data.y.push(y);
        }
    }
    
    loadExample(type) {
        switch(type) {
            case 'linear':
                this.generateLinearExample();
                break;
            case 'periodic':
                this.generatePeriodicExample();
                break;
            case 'nonlinear':
                this.generateNonlinearExample();
                break;
        }
        this.updateModel();
        this.draw();
    }
    
    generateLinearExample() {
        this.data = { x: [], y: [] };
        const slope = 0.5;
        const intercept = 1.0;
        
        this.trueFunctionType = 'linear';
        this.trueFunctionParams = { slope, intercept };
        
        for(let i = 0; i < 5; i++) {
            const x = -4 + 2 * i;
            const y = slope * x + intercept + 0.2 * (Math.random() - 0.5);
            this.data.x.push(x);
            this.data.y.push(y);
        }
        
        this.kernelType = 'rbf';
        this.lengthScale = 2.0;
        document.getElementById('kernelType').value = 'rbf';
        document.getElementById('lengthScale').value = Math.log(2.0);
        document.getElementById('lengthScale').nextElementSibling.textContent = '2.00';
    }
    
    generatePeriodicExample() {
        this.data = { x: [], y: [] };
        
        this.trueFunctionType = 'periodic';
        this.trueFunctionParams = { period: 4.0 };
        
        for(let i = 0; i < 10; i++) {
            const x = -4 + 8 * i/9;
            const y = Math.sin(2 * Math.PI * x / 4) + 0.2 * (Math.random() - 0.5);
            this.data.x.push(x);
            this.data.y.push(y);
        }
        
        this.kernelType = 'periodic';
        this.lengthScale = 1.0;
        document.getElementById('kernelType').value = 'periodic';
        document.getElementById('lengthScale').value = Math.log(1.0);
        document.getElementById('lengthScale').nextElementSibling.textContent = '1.00';
    }
    
    generateNonlinearExample() {
        this.data = { x: [], y: [] };
        
        this.trueFunctionType = 'nonlinear';
        this.trueFunctionParams = {};
        
        for(let i = 0; i < 10; i++) {
            const x = -4 + 8 * i/9;
            const y = Math.exp(-0.5 * x * x) + 0.2 * (Math.random() - 0.5);
            this.data.x.push(x);
            this.data.y.push(y);
        }
        
        this.kernelType = 'rbf';
        this.lengthScale = 1.0;
        document.getElementById('kernelType').value = 'rbf';
        document.getElementById('lengthScale').value = Math.log(1.0);
        document.getElementById('lengthScale').nextElementSibling.textContent = '1.00';
    }
    
    handleCanvasClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = -5 + 10 * (event.clientX - rect.left) / this.canvas.width;
        const y = 5 - 10 * (event.clientY - rect.top) / this.canvas.height;
        
        this.data.x.push(x);
        this.data.y.push(y);
        
        this.updateModel();
        this.draw();
    }
    
    drawTrueFunction(xTest) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.ctx.strokeStyle = '#ff6b6b';  // Coral red color for true function
        this.ctx.lineWidth = 2.5;  // Make line thicker
        this.ctx.setLineDash([6, 3]);  // Larger dash pattern
        this.ctx.beginPath();
        
        let x = 50;
        for(let i = 0; i < xTest.length; i++) {
            const trueY = this.getTrueFunctionValue(xTest[i]);
            const y = height/2 - (height - 100) * trueY / 10;
            
            if(i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
            
            x += (width - 100) / (xTest.length - 1);
        }
        
        this.ctx.stroke();
        this.ctx.setLineDash([]);  // Reset to solid line
    }
    
    getTrueFunctionValue(x) {
        if (!this.trueFunctionType) return 0;

        switch(this.trueFunctionType) {
            case 'linear':
                const {slope, intercept} = this.trueFunctionParams;
                return slope * x + intercept;
            case 'periodic':
                return Math.sin(2 * Math.PI * x / 4);
            case 'nonlinear':
                return Math.exp(-0.5 * x * x);
            default:
                return 0;
        }
    }
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const demo = new GPDemo();
}); 