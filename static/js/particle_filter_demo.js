class ParticleFilterDemo {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Environment setup
        this.bounds = {
            xMin: 0,
            xMax: this.canvas.width,
            yMin: 0,
            yMax: this.canvas.height
        };
        
        // Symmetric landmarks to create ambiguity
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.landmarks = [
            { x: w/4, y: h/4 },     // Top-left
            { x: 3*w/4, y: h/4 },   // Top-right
            { x: w/4, y: 3*h/4 },   // Bottom-left
            { x: 3*w/4, y: 3*h/4 }, // Bottom-right
            { x: w/2, y: h/2 }      // Center
        ];
        
        // Robot state - start in an ambiguous position
        this.robotPose = {
            x: this.canvas.width / 4,
            y: this.canvas.height / 4,
            theta: 0
        };
        
        // Animation state
        this.isRunning = false;
        this.animationId = null;
        this.time = 0;
        
        // Initialize particle filter
        this.initializeFilter();
        
        // Bind event handlers
        this.bindEvents();
    }
    
    initializeFilter() {
        // Create noise models with lower uncertainty
        this.motionNoise = NoiseModels.createMotionNoise(0.1, 0.05);
        this.measurementNoise = NoiseModels.createMeasurementNoise(10);
        
        // Create particle filter with 1000 particles
        this.filter = new ParticleFilter(1000, this.bounds, {
            ...this.motionNoise,
            ...this.measurementNoise
        });
    }
    
    bindEvents() {
        document.getElementById('startSimulation').addEventListener('click', () => {
            if (this.isRunning) {
                this.stop();
            } else {
                this.start();
            }
        });
        
        document.getElementById('resetSimulation').addEventListener('click', () => {
            this.reset();
        });
        
        document.getElementById('numParticles').addEventListener('input', (e) => {
            if (!this.isRunning) {
                this.initializeFilter();
                this.draw();
            }
        });
        
        document.getElementById('noiseLevel').addEventListener('input', (e) => {
            const noise = parseFloat(e.target.value) / 100;
            this.motionNoise = NoiseModels.createMotionNoise(noise, noise);
            this.measurementNoise = NoiseModels.createMeasurementNoise(noise * 50);
        });
    }
    
    start() {
        if (!this.isRunning) {
            this.isRunning = true;
            document.getElementById('startSimulation').textContent = 'Stop';
            this.animate();
        }
    }
    
    stop() {
        this.isRunning = false;
        document.getElementById('startSimulation').textContent = 'Start';
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    reset() {
        this.stop();
        this.robotPose = {
            x: this.canvas.width / 4,
            y: this.canvas.height / 4,
            theta: 0
        };
        this.initializeFilter();
        this.draw();
    }
    
    // Simulate robot motion in a figure-8 pattern
    moveRobot() {
        const scale = 100;
        this.time += 0.02;
        
        // Figure-8 pattern
        const dx = scale * 0.02 * Math.cos(this.time);
        const dy = scale * 0.01 * Math.sin(2 * this.time);
        const dtheta = 0.02 * Math.cos(this.time);
        
        // Update robot pose
        this.robotPose.x += dx;
        this.robotPose.y += dy;
        this.robotPose.theta += dtheta;
        
        // Keep robot in bounds
        this.robotPose.x = Math.max(50, Math.min(this.canvas.width - 50, this.robotPose.x));
        this.robotPose.y = Math.max(50, Math.min(this.canvas.height - 50, this.robotPose.y));
        
        // Normalize angle
        this.robotPose.theta = Math.atan2(Math.sin(this.robotPose.theta), Math.cos(this.robotPose.theta));
        
        return { dx, dy, dtheta };
    }
    
    // Get measurements to landmarks with increased noise
    getMeasurements() {
        return this.landmarks.map(landmark => {
            const dx = landmark.x - this.robotPose.x;
            const dy = landmark.y - this.robotPose.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) - this.robotPose.theta;
            
            // Add more noise to measurements
            const distNoise = this.measurementNoise.gaussian(0, this.measurementNoise.measurement * 2);
            const angleNoise = this.measurementNoise.gaussian(0, this.measurementNoise.measurement);
            
            return {
                distance: distance + distNoise,
                angle: angle + angleNoise
            };
        });
    }
    
    animate() {
        // Move robot
        const motion = this.moveRobot();
        
        // Get measurements
        const measurements = this.getMeasurements();
        
        // Update particle filter
        this.filter.predict(motion.dx, motion.dy, motion.dtheta);
        this.filter.update(measurements, this.landmarks);
        this.filter.resample();
        
        // Draw
        this.draw();
        
        // Continue animation
        if (this.isRunning) {
            this.animationId = requestAnimationFrame(() => this.animate());
        }
    }
    
    draw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw landmarks
        this.landmarks.forEach(landmark => {
            this.ctx.beginPath();
            this.ctx.arc(landmark.x, landmark.y, 8, 0, 2 * Math.PI);
            this.ctx.fillStyle = 'red';
            this.ctx.fill();
            this.ctx.strokeStyle = 'darkred';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
        
        // Draw particles with density-based opacity
        const particles = this.filter.getParticleData();
        const gridSize = 20;
        const grid = new Map();
        
        // Count particles in grid cells
        particles.forEach(particle => {
            const gridX = Math.floor(particle.x / gridSize);
            const gridY = Math.floor(particle.y / gridSize);
            const key = `${gridX},${gridY}`;
            grid.set(key, (grid.get(key) || 0) + particle.weight);
        });
        
        // Find max density for normalization
        const maxDensity = Math.max(...grid.values());
        
        // Draw particles with density-based visualization
        particles.forEach(particle => {
            const gridX = Math.floor(particle.x / gridSize);
            const gridY = Math.floor(particle.y / gridSize);
            const density = grid.get(`${gridX},${gridY}`) / maxDensity;
            
            // Draw position with density-based color
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, 3, 0, 2 * Math.PI);
            this.ctx.fillStyle = `rgba(0, 0, 255, ${Math.min(0.8, density)})`;
            this.ctx.fill();
        });

        // Calculate and draw uncertainty ellipse
        const stats = this.calculateParticleStats(particles);
        this.drawUncertaintyEllipse(stats);
        
        // Draw robot (actual position)
        this.ctx.beginPath();
        this.ctx.arc(this.robotPose.x, this.robotPose.y, 10, 0, 2 * Math.PI);
        this.ctx.fillStyle = 'green';
        this.ctx.fill();
        this.ctx.strokeStyle = 'darkgreen';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw robot orientation
        this.ctx.beginPath();
        this.ctx.moveTo(this.robotPose.x, this.robotPose.y);
        this.ctx.lineTo(
            this.robotPose.x + 20 * Math.cos(this.robotPose.theta),
            this.robotPose.y + 20 * Math.sin(this.robotPose.theta)
        );
        this.ctx.strokeStyle = 'green';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();

        // Draw convergence indicator
        const convergence = this.calculateConvergence(stats);
        this.drawConvergenceIndicator(convergence);
    }

    calculateParticleStats(particles) {
        let meanX = 0, meanY = 0, totalWeight = 0;
        
        // Calculate weighted mean
        particles.forEach(p => {
            meanX += p.x * p.weight;
            meanY += p.y * p.weight;
            totalWeight += p.weight;
        });
        meanX /= totalWeight;
        meanY /= totalWeight;
        
        // Calculate covariance
        let covXX = 0, covYY = 0, covXY = 0;
        particles.forEach(p => {
            const dx = p.x - meanX;
            const dy = p.y - meanY;
            covXX += dx * dx * p.weight;
            covYY += dy * dy * p.weight;
            covXY += dx * dy * p.weight;
        });
        covXX /= totalWeight;
        covYY /= totalWeight;
        covXY /= totalWeight;
        
        return { meanX, meanY, covXX, covYY, covXY };
    }

    drawUncertaintyEllipse(stats) {
        // Calculate eigenvalues and eigenvectors of covariance matrix
        const { meanX, meanY, covXX, covYY, covXY } = stats;
        
        // Calculate eigenvalues
        const trace = covXX + covYY;
        const det = covXX * covYY - covXY * covXY;
        const discriminant = Math.sqrt(trace * trace - 4 * det);
        const lambda1 = (trace + discriminant) / 2;  // Larger eigenvalue
        const lambda2 = (trace - discriminant) / 2;  // Smaller eigenvalue
        
        // Calculate eigenvector angle for larger eigenvalue
        const theta = Math.atan2(2 * covXY, covXX - covYY) / 2;
        
        // Draw 95% confidence ellipse (2.448 standard deviations)
        const scale = 2.448;  // 95% confidence for 2D Gaussian
        this.ctx.beginPath();
        this.ctx.ellipse(
            meanX, meanY,
            scale * Math.sqrt(Math.max(0.1, lambda1)),
            scale * Math.sqrt(Math.max(0.1, lambda2)),
            theta, 0, 2 * Math.PI
        );
        this.ctx.strokeStyle = 'rgba(0, 0, 255, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw mean position
        this.ctx.beginPath();
        this.ctx.arc(meanX, meanY, 4, 0, 2 * Math.PI);
        this.ctx.fillStyle = 'blue';
        this.ctx.fill();
    }

    calculateConvergence(stats) {
        // Calculate convergence based on covariance determinant
        const det = stats.covXX * stats.covYY - stats.covXY * stats.covXY;
        const areaMetric = Math.sqrt(Math.max(0.1, det));
        
        // Calculate normalized convergence (0 to 1)
        // Area of 100 sq pixels = perfect convergence
        // Area of 10000 sq pixels = no convergence
        const minArea = 10, maxArea = 100;
        return Math.max(0, Math.min(1, 
            (Math.log(maxArea) - Math.log(areaMetric)) / 
            (Math.log(maxArea) - Math.log(minArea))
        ));
    }

    drawConvergenceIndicator(convergence) {
        const width = 100;
        const height = 20;
        const x = 10;
        const y = 10;
        
        // Draw background
        this.ctx.fillStyle = '#eee';
        this.ctx.fillRect(x, y, width, height);
        
        // Draw convergence bar with smoother color transition
        const hue = convergence * 120;  // 0 = red, 120 = green
        this.ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
        this.ctx.fillRect(x, y, width * convergence, height);
        
        // Draw border
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x, y, width, height);
        
        // Draw label and value
        this.ctx.fillStyle = '#000';
        this.ctx.font = '12px Arial';
        this.ctx.fillText(`Convergence: ${(convergence * 100).toFixed(1)}%`, x, y + height + 15);
    }
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const demo = new ParticleFilterDemo('particleFilterCanvas');
    demo.draw();
}); 