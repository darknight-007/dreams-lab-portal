class ParameterEstimation {
    constructor() {
        this.data = [];
        this.trueSlope = 2.0;
        this.trueIntercept = 1.0;
        this.noiseLevel = 0.1;
        this.numPoints = 50;
        this.xRange = [0, 10];
        this.yRange = [-10, 10];
        
        this.initializeCanvases();
        this.initializeControls();
        this.setupEventListeners();
        this.setupTooltips();
    }

    initializeCanvases() {
        this.plotCanvas = document.getElementById('plotCanvas');
        this.plotCtx = this.plotCanvas.getContext('2d');
        
        // Make canvas responsive
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        const container = this.plotCanvas.parentElement;
        const width = container.clientWidth;
        this.plotCanvas.width = width;
        this.plotCanvas.height = Math.min(width * 0.6, 500);
        this.updateVisualization();
    }

    initializeControls() {
        this.slopeInput = document.getElementById('slope');
        this.interceptInput = document.getElementById('intercept');
        this.noiseLevelInput = document.getElementById('noiseLevel');
        this.numPointsInput = document.getElementById('numPoints');
    }

    setupEventListeners() {
        // Parameter controls with debouncing
        const updateWithDebounce = this.debounce(() => this.generateData(), 50);

        this.slopeInput.addEventListener('input', () => {
            this.trueSlope = parseFloat(this.slopeInput.value);
            this.slopeInput.nextElementSibling.textContent = this.trueSlope.toFixed(2);
            updateWithDebounce();
        });

        this.interceptInput.addEventListener('input', () => {
            this.trueIntercept = parseFloat(this.interceptInput.value);
            this.interceptInput.nextElementSibling.textContent = this.trueIntercept.toFixed(2);
            updateWithDebounce();
        });

        this.noiseLevelInput.addEventListener('input', () => {
            this.noiseLevel = Math.pow(10, parseFloat(this.noiseLevelInput.value));
            this.noiseLevelInput.nextElementSibling.textContent = this.noiseLevel.toFixed(3);
            updateWithDebounce();
        });

        this.numPointsInput.addEventListener('input', () => {
            this.numPoints = parseInt(this.numPointsInput.value);
            this.numPointsInput.nextElementSibling.textContent = this.numPoints;
            this.generateData();
        });

        // Generate data button
        document.getElementById('generateData').addEventListener('click', () => this.generateData());

        // Canvas interaction
        this.plotCanvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.plotCanvas.addEventListener('mouseout', () => this.hideTooltip());
    }

    setupTooltips() {
        this.tooltip = document.getElementById('tooltip');
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    handleMouseMove(event) {
        const rect = this.plotCanvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (this.plotCanvas.width / rect.width);
        const y = (event.clientY - rect.top) * (this.plotCanvas.height / rect.height);
        
        // Convert to data coordinates
        const dataX = this.pixelToDataX(x);
        const dataY = this.pixelToDataY(y);
        
        // Find nearest point
        const nearestPoint = this.findNearestPoint(dataX, dataY);
        if (nearestPoint) {
            this.showTooltip(event.pageX, event.pageY, nearestPoint);
        } else {
            this.hideTooltip();
        }
    }

    findNearestPoint(x, y) {
        const threshold = 0.5;
        let nearest = null;
        let minDist = Infinity;
        
        for (const point of this.data) {
            const dist = Math.sqrt(Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2));
            if (dist < minDist && dist < threshold) {
                minDist = dist;
                nearest = point;
            }
        }
        
        return nearest;
    }

    showTooltip(x, y, point) {
        this.tooltip.style.display = 'block';
        this.tooltip.style.left = (x + 10) + 'px';
        this.tooltip.style.top = (y + 10) + 'px';
        this.tooltip.textContent = `(${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
    }

    hideTooltip() {
        this.tooltip.style.display = 'none';
    }

    generateData() {
        this.data = [];
        
        // Generate x values between 0 and 10
        for (let i = 0; i < this.numPoints; i++) {
            const x = i * (this.xRange[1] - this.xRange[0]) / (this.numPoints - 1) + this.xRange[0];
            const trueY = this.trueSlope * x + this.trueIntercept;
            // Use Box-Muller transform for better normal distribution
            const noise = this.noiseLevel * this.generateGaussianNoise();
            const y = trueY + noise;
            this.data.push({ x, y });
        }
        
        this.updateVisualization();
    }

    generateGaussianNoise() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    leastSquares() {
        if (this.data.length < 2) return null;

        const n = this.data.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for (const point of this.data) {
            sumX += point.x;
            sumY += point.y;
            sumXY += point.x * point.y;
            sumX2 += point.x * point.x;
        }
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Calculate R-squared
        const yMean = sumY / n;
        let ssTotal = 0, ssResidual = 0;
        
        for (const point of this.data) {
            const yPred = slope * point.x + intercept;
            ssTotal += Math.pow(point.y - yMean, 2);
            ssResidual += Math.pow(point.y - yPred, 2);
        }
        
        const rSquared = 1 - (ssResidual / ssTotal);
        
        return { slope, intercept, rSquared };
    }

    dataToPixelX(x) {
        const margin = 50;
        return margin + (x - this.xRange[0]) * (this.plotCanvas.width - 2 * margin) / (this.xRange[1] - this.xRange[0]);
    }

    dataToPixelY(y) {
        const margin = 50;
        return this.plotCanvas.height - margin - (y - this.yRange[0]) * (this.plotCanvas.height - 2 * margin) / (this.yRange[1] - this.yRange[0]);
    }

    pixelToDataX(px) {
        const margin = 50;
        return this.xRange[0] + (px - margin) * (this.xRange[1] - this.xRange[0]) / (this.plotCanvas.width - 2 * margin);
    }

    pixelToDataY(py) {
        const margin = 50;
        return this.yRange[0] + (this.plotCanvas.height - margin - py) * (this.yRange[1] - this.yRange[0]) / (this.plotCanvas.height - 2 * margin);
    }

    updateVisualization() {
        const ctx = this.plotCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and axes
        this.drawGridAndAxes(ctx);
        
        // Draw data points
        ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
        this.data.forEach(point => {
            ctx.beginPath();
            ctx.arc(
                this.dataToPixelX(point.x),
                this.dataToPixelY(point.y),
                4,
                0,
                2 * Math.PI
            );
            ctx.fill();
        });
        
        // Draw true line
        this.drawLine(ctx, this.trueSlope, this.trueIntercept, 'green', 'True Line');
        
        // Draw fitted line
        const fit = this.leastSquares();
        if (fit) {
            this.drawLine(ctx, fit.slope, fit.intercept, 'red', 'Fitted Line');
            
            // Display parameters
            ctx.fillStyle = 'black';
            ctx.font = '14px Arial';
            ctx.fillText(`True: y = ${this.trueSlope.toFixed(2)}x + ${this.trueIntercept.toFixed(2)}`, 60, 30);
            ctx.fillText(`Fitted: y = ${fit.slope.toFixed(2)}x + ${fit.intercept.toFixed(2)}`, 60, 50);
            ctx.fillText(`R² = ${fit.rSquared.toFixed(4)}`, 60, 70);
        }
    }

    drawGridAndAxes(ctx) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        // Draw grid
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 1;
        
        // Vertical grid lines
        for (let x = this.xRange[0]; x <= this.xRange[1]; x++) {
            const px = this.dataToPixelX(x);
            ctx.beginPath();
            ctx.moveTo(px, 50);
            ctx.lineTo(px, height - 50);
            ctx.stroke();
            
            // X-axis labels
            ctx.fillStyle = 'black';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(x.toString(), px, height - 30);
        }
        
        // Horizontal grid lines
        for (let y = Math.ceil(this.yRange[0]); y <= this.yRange[1]; y++) {
            const py = this.dataToPixelY(y);
            ctx.beginPath();
            ctx.moveTo(50, py);
            ctx.lineTo(width - 50, py);
            ctx.stroke();
            
            // Y-axis labels
            ctx.fillStyle = 'black';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.fillText(y.toString(), 40, py + 4);
        }
        
        // Draw axes
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(50, this.dataToPixelY(0));
        ctx.lineTo(width - 50, this.dataToPixelY(0));
        ctx.stroke();
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(this.dataToPixelX(0), 50);
        ctx.lineTo(this.dataToPixelX(0), height - 50);
        ctx.stroke();
    }

    drawLine(ctx, slope, intercept, color, label) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.dataToPixelX(this.xRange[0]), this.dataToPixelY(slope * this.xRange[0] + intercept));
        ctx.lineTo(this.dataToPixelX(this.xRange[1]), this.dataToPixelY(slope * this.xRange[1] + intercept));
        ctx.stroke();
        
        // Add legend
        const legendY = color === 'green' ? 90 : 110;
        ctx.fillStyle = color;
        ctx.fillRect(60, legendY - 8, 20, 2);
        ctx.fillStyle = 'black';
        ctx.fillText(label, 90, legendY);
    }
} 