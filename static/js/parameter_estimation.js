class ParameterEstimation {
    constructor() {
        this.data = [];
        this.polynomialOrder = 3;
        this.priorVariance = 1.0;
        this.noiseLevel = 0.01;
        this.viewType = 'function';
        this.showLS = true;
        this.showBayes = true;
        this.showUncertainty = true;
        
        this.initializeCanvases();
        this.initializeControls();
        this.setupEventListeners();
    }

    initializeCanvases() {
        this.functionCanvas = document.getElementById('functionCanvas');
        this.weightCanvas = document.getElementById('weightCanvas');
        this.evolutionCanvas = document.getElementById('evolutionCanvas');
        this.comparisonCanvas = document.getElementById('comparisonCanvas');
        
        this.functionCtx = this.functionCanvas.getContext('2d');
        this.weightCtx = this.weightCanvas.getContext('2d');
        this.evolutionCtx = this.evolutionCanvas.getContext('2d');
        this.comparisonCtx = this.comparisonCanvas.getContext('2d');
    }

    initializeControls() {
        this.polynomialOrderInput = document.getElementById('polynomialOrder');
        this.priorVarianceInput = document.getElementById('priorVariance');
        this.noiseLevelInput = document.getElementById('noiseLevel');
        
        this.viewTypeInputs = document.getElementsByName('viewType');
        this.showLSCheckbox = document.getElementById('showLS');
        this.showBayesCheckbox = document.getElementById('showBayes');
        this.showUncertaintyCheckbox = document.getElementById('showUncertainty');
    }

    setupEventListeners() {
        // Parameter controls
        this.polynomialOrderInput.addEventListener('input', () => {
            this.polynomialOrder = parseInt(this.polynomialOrderInput.value);
            this.polynomialOrderInput.nextElementSibling.textContent = this.polynomialOrder;
            this.updateVisualization();
        });

        this.priorVarianceInput.addEventListener('input', () => {
            this.priorVariance = Math.pow(10, parseFloat(this.priorVarianceInput.value));
            this.priorVarianceInput.nextElementSibling.textContent = this.priorVariance.toFixed(2);
            this.updateVisualization();
        });

        this.noiseLevelInput.addEventListener('input', () => {
            this.noiseLevel = Math.pow(10, parseFloat(this.noiseLevelInput.value));
            this.noiseLevelInput.nextElementSibling.textContent = this.noiseLevel.toFixed(4);
            this.updateVisualization();
        });

        // View controls
        this.viewTypeInputs.forEach(input => {
            input.addEventListener('change', () => {
                this.viewType = input.value;
                this.updateVisualization();
            });
        });

        this.showLSCheckbox.addEventListener('change', () => {
            this.showLS = this.showLSCheckbox.checked;
            this.updateVisualization();
        });

        this.showBayesCheckbox.addEventListener('change', () => {
            this.showBayes = this.showBayesCheckbox.checked;
            this.updateVisualization();
        });

        this.showUncertaintyCheckbox.addEventListener('change', () => {
            this.showUncertainty = this.showUncertaintyCheckbox.checked;
            this.updateVisualization();
        });

        // Data controls
        document.getElementById('generateData').addEventListener('click', () => this.generateRandomData());
        document.getElementById('clearData').addEventListener('click', () => this.clearData());
        document.getElementById('optimizeParams').addEventListener('click', () => this.optimizeParameters());

        // Example buttons
        document.querySelectorAll('.example-btn').forEach(button => {
            button.addEventListener('click', () => this.loadExample(button.dataset.example));
        });

        // Canvas click handlers
        this.functionCanvas.addEventListener('click', (e) => this.handleCanvasClick(e));
    }

    handleCanvasClick(event) {
        const rect = this.functionCanvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width;
        const y = (event.clientY - rect.top) / rect.height;
        
        // Transform to data coordinates
        const dataX = x * 10;
        const dataY = (1 - y) * 10;
        
        this.data.push({ x: dataX, y: dataY });
        this.updateVisualization();
    }

    generateRandomData() {
        this.data = [];
        const n = 20;
        
        for (let i = 0; i < n; i++) {
            const x = Math.random() * 10;
            let y;
            
            // Generate y based on current example or default to quadratic
            if (this.currentExample === 'linear') {
                y = 2 * x + 1 + this.generateNoise();
            } else if (this.currentExample === 'sinusoidal') {
                y = 5 * Math.sin(x) + this.generateNoise();
            } else {
                // Default quadratic
                y = 0.5 * x * x - 2 * x + 3 + this.generateNoise();
            }
            
            this.data.push({ x, y });
        }
        
        this.updateVisualization();
    }

    generateNoise() {
        return this.noiseLevel * (Math.random() + Math.random() + Math.random() - 1.5) * 3;
    }

    clearData() {
        this.data = [];
        this.updateVisualization();
    }

    optimizeParameters() {
        if (this.data.length < this.polynomialOrder + 1) {
            alert('Need more data points than polynomial coefficients');
            return;
        }

        // Implement cross-validation or other parameter optimization
        this.updateVisualization();
    }

    loadExample(example) {
        this.currentExample = example;
        this.clearData();
        
        switch (example) {
            case 'linear':
                this.polynomialOrder = 1;
                this.polynomialOrderInput.value = 1;
                break;
            case 'quadratic':
                this.polynomialOrder = 2;
                this.polynomialOrderInput.value = 2;
                break;
            case 'sinusoidal':
                this.polynomialOrder = 5;
                this.polynomialOrderInput.value = 5;
                break;
            case 'outliers':
                this.polynomialOrder = 2;
                this.polynomialOrderInput.value = 2;
                this.noiseLevel = 0.5;
                this.noiseLevelInput.value = Math.log10(0.5);
                break;
        }
        
        this.generateRandomData();
    }

    computeDesignMatrix(x, order) {
        const X = [];
        for (let i = 0; i < x.length; i++) {
            const row = [];
            for (let j = 0; j <= order; j++) {
                row.push(Math.pow(x[i], j));
            }
            X.push(row);
        }
        return X;
    }

    leastSquares(X, y) {
        const XT = math.transpose(X);
        const XTX = math.multiply(XT, X);
        const XTy = math.multiply(XT, y);
        return math.multiply(math.inv(XTX), XTy);
    }

    bayesianRegression(X, y) {
        const n = X.length;
        const d = X[0].length;
        
        // Prior precision matrix (inverse covariance)
        const alpha = 1 / this.priorVariance;
        const priorPrecision = math.multiply(math.identity(d), alpha);
        
        // Data precision (inverse noise variance)
        const beta = 1 / (this.noiseLevel * this.noiseLevel);
        
        // Posterior precision matrix
        const XT = math.transpose(X);
        const XTX = math.multiply(XT, X);
        const posteriorPrecision = math.add(
            priorPrecision,
            math.multiply(XTX, beta)
        );
        
        // Posterior mean
        const XTy = math.multiply(XT, y);
        const posteriorMean = math.multiply(
            math.inv(posteriorPrecision),
            math.multiply(XTy, beta)
        );
        
        return {
            mean: posteriorMean,
            covariance: math.inv(posteriorPrecision)
        };
    }

    updateVisualization() {
        this.clearCanvases();
        
        if (this.data.length === 0) {
            return;
        }

        const x = this.data.map(d => d.x);
        const y = this.data.map(d => d.y);
        const X = this.computeDesignMatrix(x, this.polynomialOrder);
        
        // Compute LS and Bayesian estimates
        const lsEstimate = this.leastSquares(X, y);
        const bayesEstimate = this.bayesianRegression(X, y);
        
        if (this.viewType === 'function') {
            this.drawFunctionSpace(X, y, lsEstimate, bayesEstimate);
        } else {
            this.drawWeightSpace(lsEstimate, bayesEstimate);
        }
        
        this.drawParameterEvolution();
        this.drawComparison(lsEstimate, bayesEstimate);
    }

    clearCanvases() {
        [this.functionCtx, this.weightCtx, this.evolutionCtx, this.comparisonCtx].forEach(ctx => {
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            this.drawGrid(ctx);
        });
    }

    drawGrid(ctx) {
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 1;
        
        // Draw grid lines
        for (let i = 0; i <= 10; i++) {
            const x = i * ctx.canvas.width / 10;
            const y = i * ctx.canvas.height / 10;
            
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, ctx.canvas.height);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(ctx.canvas.width, y);
            ctx.stroke();
        }
    }

    drawFunctionSpace(X, y, lsEstimate, bayesEstimate) {
        const ctx = this.functionCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        // Draw data points
        ctx.fillStyle = 'black';
        this.data.forEach(point => {
            ctx.beginPath();
            ctx.arc(
                point.x * width / 10,
                height - point.y * height / 10,
                3,
                0,
                2 * Math.PI
            );
            ctx.fill();
        });
        
        // Draw predictions
        const testX = Array.from({ length: 100 }, (_, i) => i * 10 / 99);
        const testDesign = this.computeDesignMatrix(testX, this.polynomialOrder);
        
        if (this.showLS) {
            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.beginPath();
            testX.forEach((x, i) => {
                const pred = math.dot(testDesign[i], lsEstimate);
                const px = x * width / 10;
                const py = height - pred * height / 10;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.stroke();
        }
        
        if (this.showBayes) {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.beginPath();
            testX.forEach((x, i) => {
                const pred = math.dot(testDesign[i], bayesEstimate.mean);
                const px = x * width / 10;
                const py = height - pred * height / 10;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.stroke();
            
            if (this.showUncertainty) {
                ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
                testX.forEach((x, i) => {
                    const designVector = testDesign[i];
                    const variance = math.dot(
                        math.multiply(designVector, bayesEstimate.covariance),
                        designVector
                    ) + this.noiseLevel * this.noiseLevel;
                    const std = Math.sqrt(variance);
                    
                    const pred = math.dot(designVector, bayesEstimate.mean);
                    const px = x * width / 10;
                    const py = height - pred * height / 10;
                    
                    ctx.fillRect(
                        px - 1,
                        height - (pred + 2 * std) * height / 10,
                        2,
                        4 * std * height / 10
                    );
                });
            }
        }
    }

    drawWeightSpace(lsEstimate, bayesEstimate) {
        if (this.polynomialOrder > 2) {
            this.weightCtx.font = '14px Arial';
            this.weightCtx.fillStyle = 'black';
            this.weightCtx.fillText(
                'Weight space visualization only available for polynomial order ≤ 2',
                20,
                this.weightCanvas.height / 2
            );
            return;
        }
        
        const ctx = this.weightCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        // Draw coordinate axes
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.moveTo(width / 2, 0);
        ctx.lineTo(width / 2, height);
        ctx.stroke();
        
        if (this.showLS) {
            ctx.fillStyle = 'blue';
            const lsX = (lsEstimate[0] + 5) * width / 10;
            const lsY = height - (lsEstimate[1] + 5) * height / 10;
            ctx.beginPath();
            ctx.arc(lsX, lsY, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        if (this.showBayes) {
            ctx.fillStyle = 'red';
            const bayesX = (bayesEstimate.mean[0] + 5) * width / 10;
            const bayesY = height - (bayesEstimate.mean[1] + 5) * height / 10;
            ctx.beginPath();
            ctx.arc(bayesX, bayesY, 5, 0, 2 * Math.PI);
            ctx.fill();
            
            if (this.showUncertainty) {
                this.drawUncertaintyEllipse(
                    ctx,
                    bayesEstimate.mean,
                    bayesEstimate.covariance,
                    width,
                    height
                );
            }
        }
    }

    drawUncertaintyEllipse(ctx, mean, covariance, width, height) {
        // Compute eigendecomposition
        const [eigenvalues, eigenvectors] = math.eigs(covariance);
        
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
        ctx.lineWidth = 1;
        
        // Draw 95% confidence ellipse
        const scale = Math.sqrt(5.991); // 95% confidence for 2D Gaussian
        const steps = 50;
        
        ctx.beginPath();
        for (let i = 0; i <= steps; i++) {
            const angle = (2 * Math.PI * i) / steps;
            const x = scale * Math.cos(angle);
            const y = scale * Math.sin(angle);
            
            // Transform point by eigenvectors and eigenvalues
            const transformedX = 
                mean[0] + 
                Math.sqrt(eigenvalues.x[0]) * eigenvectors[0][0] * x +
                Math.sqrt(eigenvalues.x[1]) * eigenvectors[0][1] * y;
            const transformedY = 
                mean[1] + 
                Math.sqrt(eigenvalues.x[0]) * eigenvectors[1][0] * x +
                Math.sqrt(eigenvalues.x[1]) * eigenvectors[1][1] * y;
            
            const px = (transformedX + 5) * width / 10;
            const py = height - (transformedY + 5) * height / 10;
            
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    drawParameterEvolution() {
        // Implement parameter evolution visualization
        // This could show how parameters change as more data points are added
    }

    drawComparison(lsEstimate, bayesEstimate) {
        const ctx = this.comparisonCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        // Compute metrics
        const metrics = this.computeMetrics(lsEstimate, bayesEstimate);
        
        // Draw bar chart
        const barWidth = width / 6;
        const spacing = width / 12;
        
        // MSE bars
        if (this.showLS) {
            ctx.fillStyle = 'blue';
            const lsHeight = metrics.lsMSE * height / 2;
            ctx.fillRect(spacing, height - lsHeight, barWidth, lsHeight);
        }
        
        if (this.showBayes) {
            ctx.fillStyle = 'red';
            const bayesHeight = metrics.bayesMSE * height / 2;
            ctx.fillRect(2 * spacing + barWidth, height - bayesHeight, barWidth, bayesHeight);
        }
        
        // Complexity penalty bars
        if (this.showLS) {
            ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
            const lsHeight = metrics.lsComplexity * height / 2;
            ctx.fillRect(4 * spacing + 2 * barWidth, height - lsHeight, barWidth, lsHeight);
        }
        
        if (this.showBayes) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
            const bayesHeight = metrics.bayesComplexity * height / 2;
            ctx.fillRect(5 * spacing + 3 * barWidth, height - bayesHeight, barWidth, bayesHeight);
        }
        
        // Labels
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        
        ctx.fillText('MSE', 2 * spacing + barWidth, height - 10);
        ctx.fillText('Complexity', 5 * spacing + 3 * barWidth, height - 10);
    }

    computeMetrics(lsEstimate, bayesEstimate) {
        // Compute MSE
        const predictions = this.data.map(point => {
            const design = Array.from(
                { length: this.polynomialOrder + 1 },
                (_, i) => Math.pow(point.x, i)
            );
            return {
                ls: math.dot(design, lsEstimate),
                bayes: math.dot(design, bayesEstimate.mean)
            };
        });
        
        const lsMSE = predictions.reduce(
            (sum, pred, i) => sum + Math.pow(pred.ls - this.data[i].y, 2),
            0
        ) / this.data.length;
        
        const bayesMSE = predictions.reduce(
            (sum, pred, i) => sum + Math.pow(pred.bayes - this.data[i].y, 2),
            0
        ) / this.data.length;
        
        // Compute complexity penalties
        const lsComplexity = math.norm(lsEstimate);
        const bayesComplexity = math.norm(bayesEstimate.mean);
        
        return {
            lsMSE: Math.min(lsMSE, 2), // Cap for visualization
            bayesMSE: Math.min(bayesMSE, 2),
            lsComplexity: Math.min(lsComplexity, 2),
            bayesComplexity: Math.min(bayesComplexity, 2)
        };
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.paramEstimation = new ParameterEstimation();
}); 