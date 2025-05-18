class SLAMBuddy {
    constructor() {
        // Initialize state
        this.robot = {
            x: 0,
            y: 0,
            theta: 0,
            covariance: [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
        };
        this.landmarks = [];
        this.particles = [];
        this.measurements = [];
        this.groundTruth = {
            robot: { x: 0, y: 0, theta: 0 },
            landmarks: []
        };
        this.path = [];
        this.isRunning = false;
        this.isDragging = false;
        
        // Initialize parameters
        this.motionNoise = 0.2;
        this.measurementNoise = 0.1;
        this.numParticles = 100;
        this.algorithm = 'ekf';
        
        // Setup canvas
        this.canvas = document.getElementById('slam-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();
        
        // Bind event listeners
        window.addEventListener('resize', () => this.resizeCanvas());
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.handleMouseUp());
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Start animation loop
        this.animate();
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.draw();
    }

    reset() {
        this.robot = {
            x: 0,
            y: 0,
            theta: 0,
            covariance: [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
        };
        this.landmarks = [];
        this.particles = [];
        this.measurements = [];
        this.groundTruth = {
            robot: { x: 0, y: 0, theta: 0 },
            landmarks: []
        };
        this.path = [];
        this.isRunning = false;
        this.draw();
    }

    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Left click: Add landmark
        if (e.button === 0) {
            this.landmarks.push({
                x,
                y,
                id: this.landmarks.length,
                covariance: [[0.1, 0], [0, 0.1]]
            });
            this.groundTruth.landmarks.push({ x, y });
        }
        // Right click: Set/drag robot
        else if (e.button === 2) {
            this.isDragging = true;
            this.robot.x = x;
            this.robot.y = y;
            this.groundTruth.robot.x = x;
            this.groundTruth.robot.y = y;
            this.path = [{ x, y }];
        }
        
        this.draw();
    }

    handleMouseMove(e) {
        if (!this.isDragging) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Update robot position
        this.robot.x = x;
        this.robot.y = y;
        this.groundTruth.robot.x = x;
        this.groundTruth.robot.y = y;
        
        // Update path
        this.path.push({ x, y });
        
        // Take measurements
        this.updateMeasurements();
        
        this.draw();
    }

    handleMouseUp() {
        this.isDragging = false;
    }

    updateMeasurements() {
        this.measurements = [];
        for (const landmark of this.landmarks) {
            const dx = landmark.x - this.robot.x;
            const dy = landmark.y - this.robot.y;
            const range = Math.sqrt(dx * dx + dy * dy);
            const bearing = Math.atan2(dy, dx) - this.robot.theta;
            
            // Add noise to measurements
            const noisyRange = range + (Math.random() - 0.5) * this.measurementNoise * range;
            const noisyBearing = bearing + (Math.random() - 0.5) * this.measurementNoise;
            
            this.measurements.push({
                landmarkId: landmark.id,
                range: noisyRange,
                bearing: noisyBearing
            });
        }
    }

    animate() {
        if (this.isRunning) {
            this.step();
        }
        requestAnimationFrame(() => this.animate());
    }

    step() {
        switch (this.algorithm) {
            case 'ekf':
                this.stepEKF();
                break;
            case 'fastslam':
                this.stepFastSLAM();
                break;
            case 'graph':
                this.stepGraphSLAM();
                break;
        }
        this.updateMeasurements();
        this.draw();
    }

    stepEKF() {
        // Predict step
        this.robot.covariance = this.addMotionNoise(this.robot.covariance);
        
        // Update step for each measurement
        for (const measurement of this.measurements) {
            const landmark = this.landmarks[measurement.landmarkId];
            if (!landmark) continue;
            
            // Compute Kalman gain and update state
            this.updateEKF(landmark, measurement);
        }
    }

    stepFastSLAM() {
        // Implement FastSLAM particle filter update
        if (this.particles.length === 0) {
            this.initializeParticles();
        }
        
        // Predict particles
        this.particles = this.particles.map(particle => this.predictParticle(particle));
        
        // Update particles with measurements
        this.particles = this.updateParticles(this.measurements);
        
        // Resample particles
        this.particles = this.resampleParticles();
    }

    stepGraphSLAM() {
        // Implement Graph SLAM optimization
        // This is a simplified version that only optimizes the latest pose
        if (this.path.length < 2) return;
        
        const prevPose = this.path[this.path.length - 2];
        const currPose = this.path[this.path.length - 1];
        
        // Add motion constraints
        this.addMotionConstraint(prevPose, currPose);
        
        // Add measurement constraints
        for (const measurement of this.measurements) {
            this.addMeasurementConstraint(currPose, measurement);
        }
        
        // Optimize (simplified)
        this.optimizeGraphSLAM();
    }

    draw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        this.drawGrid();
        
        // Draw ground truth if enabled
        if (document.getElementById('show-ground-truth')?.checked) {
            this.drawGroundTruth();
        }
        
        // Draw landmarks and their uncertainties
        this.drawLandmarks();
        
        // Draw particles if enabled
        if (document.getElementById('show-particles')?.checked && this.algorithm === 'fastslam') {
            this.drawParticles();
        }
        
        // Draw measurements if enabled
        if (document.getElementById('show-measurements')?.checked) {
            this.drawMeasurements();
        }
        
        // Draw robot and its uncertainty
        this.drawRobot();
        
        // Draw path
        this.drawPath();
    }

    drawGrid() {
        const gridSize = 50;
        this.ctx.strokeStyle = '#eee';
        this.ctx.lineWidth = 1;
        
        // Draw vertical lines
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Draw horizontal lines
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    drawRobot() {
        // Draw robot position
        this.ctx.fillStyle = '#4CAF50';
        this.ctx.beginPath();
        this.ctx.arc(this.robot.x, this.robot.y, 10, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw robot heading
        this.ctx.strokeStyle = '#4CAF50';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.robot.x, this.robot.y);
        this.ctx.lineTo(
            this.robot.x + 20 * Math.cos(this.robot.theta),
            this.robot.y + 20 * Math.sin(this.robot.theta)
        );
        this.ctx.stroke();
        
        // Draw uncertainty ellipse if enabled
        if (document.getElementById('show-uncertainty')?.checked) {
            this.drawUncertaintyEllipse(
                this.robot.x,
                this.robot.y,
                this.robot.covariance[0][0],
                this.robot.covariance[1][1],
                '#4CAF5044'
            );
        }
    }

    drawLandmarks() {
        this.ctx.fillStyle = '#f44336';
        for (const landmark of this.landmarks) {
            // Draw landmark
            this.ctx.beginPath();
            this.ctx.arc(landmark.x, landmark.y, 5, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw uncertainty if enabled
            if (document.getElementById('show-uncertainty')?.checked) {
                this.drawUncertaintyEllipse(
                    landmark.x,
                    landmark.y,
                    landmark.covariance[0][0],
                    landmark.covariance[1][1],
                    '#f4433644'
                );
            }
        }
    }

    drawParticles() {
        this.ctx.fillStyle = '#2196F3';
        for (const particle of this.particles) {
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, 2, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    drawMeasurements() {
        this.ctx.strokeStyle = '#FF980066';
        this.ctx.lineWidth = 1;
        for (const measurement of this.measurements) {
            const landmark = this.landmarks[measurement.landmarkId];
            if (!landmark) continue;
            
            this.ctx.beginPath();
            this.ctx.moveTo(this.robot.x, this.robot.y);
            this.ctx.lineTo(landmark.x, landmark.y);
            this.ctx.stroke();
        }
    }

    drawPath() {
        if (this.path.length < 2) return;
        
        this.ctx.strokeStyle = '#4CAF5088';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.path[0].x, this.path[0].y);
        
        for (let i = 1; i < this.path.length; i++) {
            this.ctx.lineTo(this.path[i].x, this.path[i].y);
        }
        
        this.ctx.stroke();
    }

    drawGroundTruth() {
        // Draw ground truth robot position
        this.ctx.strokeStyle = '#4CAF5044';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(this.groundTruth.robot.x, this.groundTruth.robot.y, 12, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Draw ground truth landmarks
        this.ctx.strokeStyle = '#f4433644';
        for (const landmark of this.groundTruth.landmarks) {
            this.ctx.beginPath();
            this.ctx.arc(landmark.x, landmark.y, 7, 0, Math.PI * 2);
            this.ctx.stroke();
        }
    }

    drawUncertaintyEllipse(x, y, varX, varY, color) {
        const scale = 2; // 95% confidence interval
        const width = Math.sqrt(varX) * scale;
        const height = Math.sqrt(varY) * scale;
        
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.ellipse(x, y, width, height, 0, 0, Math.PI * 2);
        this.ctx.fill();
    }

    addRandomLandmarks() {
        const numLandmarks = 10;
        for (let i = 0; i < numLandmarks; i++) {
            const x = Math.random() * this.canvas.width;
            const y = Math.random() * this.canvas.height;
            
            this.landmarks.push({
                x,
                y,
                id: this.landmarks.length,
                covariance: [[0.1, 0], [0, 0.1]]
            });
            
            this.groundTruth.landmarks.push({ x, y });
        }
        this.draw();
    }

    generateLoopPath() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const radius = Math.min(this.canvas.width, this.canvas.height) / 4;
        
        this.path = [];
        for (let angle = 0; angle <= Math.PI * 2; angle += 0.1) {
            this.path.push({
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle)
            });
        }
        
        // Set robot to start of path
        if (this.path.length > 0) {
            this.robot.x = this.path[0].x;
            this.robot.y = this.path[0].y;
            this.groundTruth.robot.x = this.path[0].x;
            this.groundTruth.robot.y = this.path[0].y;
        }
        
        this.draw();
    }

    // Helper methods for SLAM algorithms
    addMotionNoise(covariance) {
        const noise = this.motionNoise * this.motionNoise;
        return covariance.map((row, i) =>
            row.map((val, j) => val + (i === j ? noise : 0))
        );
    }

    updateEKF(landmark, measurement) {
        // Simplified EKF update
        const dx = landmark.x - this.robot.x;
        const dy = landmark.y - this.robot.y;
        const q = dx * dx + dy * dy;
        const sqrt_q = Math.sqrt(q);
        
        // Jacobian
        const H = [
            [-dx / sqrt_q, -dy / sqrt_q],
            [dy / q, -dx / q]
        ];
        
        // Measurement noise
        const R = [
            [this.measurementNoise * this.measurementNoise, 0],
            [0, this.measurementNoise * this.measurementNoise]
        ];
        
        // Kalman gain
        const K = this.computeKalmanGain(H, R);
        
        // Update state
        const innovation = [
            measurement.range - sqrt_q,
            measurement.bearing - Math.atan2(dy, dx)
        ];
        
        this.robot.x += K[0][0] * innovation[0] + K[0][1] * innovation[1];
        this.robot.y += K[1][0] * innovation[0] + K[1][1] * innovation[1];
        
        // Update covariance
        this.robot.covariance = this.updateCovariance(K, H);
    }

    computeKalmanGain(H, R) {
        // Simplified Kalman gain computation
        return [[0.1, 0], [0, 0.1]]; // Placeholder
    }

    updateCovariance(K, H) {
        // Simplified covariance update
        return this.robot.covariance.map(row => row.map(val => val * 0.9));
    }

    initializeParticles() {
        this.particles = [];
        for (let i = 0; i < this.numParticles; i++) {
            this.particles.push({
                x: this.robot.x + (Math.random() - 0.5) * 20,
                y: this.robot.y + (Math.random() - 0.5) * 20,
                theta: this.robot.theta + (Math.random() - 0.5) * 0.1,
                weight: 1 / this.numParticles
            });
        }
    }

    predictParticle(particle) {
        return {
            ...particle,
            x: particle.x + (Math.random() - 0.5) * this.motionNoise,
            y: particle.y + (Math.random() - 0.5) * this.motionNoise,
            theta: particle.theta + (Math.random() - 0.5) * this.motionNoise * 0.1
        };
    }

    updateParticles(measurements) {
        return this.particles.map(particle => {
            let weight = particle.weight;
            
            for (const measurement of measurements) {
                const landmark = this.landmarks[measurement.landmarkId];
                if (!landmark) continue;
                
                const dx = landmark.x - particle.x;
                const dy = landmark.y - particle.y;
                const expectedRange = Math.sqrt(dx * dx + dy * dy);
                const expectedBearing = Math.atan2(dy, dx) - particle.theta;
                
                // Update weight based on measurement likelihood
                const rangeError = measurement.range - expectedRange;
                const bearingError = measurement.bearing - expectedBearing;
                weight *= Math.exp(-(rangeError * rangeError + bearingError * bearingError) / 
                    (2 * this.measurementNoise * this.measurementNoise));
            }
            
            return { ...particle, weight };
        });
    }

    resampleParticles() {
        // Normalize weights
        const totalWeight = this.particles.reduce((sum, p) => sum + p.weight, 0);
        const normalizedParticles = this.particles.map(p => ({
            ...p,
            weight: p.weight / totalWeight
        }));
        
        // Resample
        const newParticles = [];
        for (let i = 0; i < this.numParticles; i++) {
            const r = Math.random();
            let sum = 0;
            for (const particle of normalizedParticles) {
                sum += particle.weight;
                if (sum >= r) {
                    newParticles.push({
                        ...particle,
                        weight: 1 / this.numParticles
                    });
                    break;
                }
            }
        }
        
        return newParticles;
    }

    addMotionConstraint(prevPose, currPose) {
        // Add motion constraint to graph (simplified)
    }

    addMeasurementConstraint(pose, measurement) {
        // Add measurement constraint to graph (simplified)
    }

    optimizeGraphSLAM() {
        // Optimize graph (simplified)
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const slam = new SLAMBuddy();
    
    // Add event listeners for controls
    document.getElementById('start-slam').addEventListener('click', () => {
        slam.isRunning = !slam.isRunning;
        document.getElementById('start-slam').textContent = 
            slam.isRunning ? 'Pause SLAM' : 'Start SLAM';
    });
    
    document.getElementById('reset-slam').addEventListener('click', () => slam.reset());
    document.getElementById('add-random-landmarks').addEventListener('click', () => slam.addRandomLandmarks());
    document.getElementById('generate-loop').addEventListener('click', () => slam.generateLoopPath());
    
    // Add listener for algorithm selection
    document.getElementById('slam-algorithm').addEventListener('change', (e) => {
        slam.algorithm = e.target.value;
        slam.reset();
    });
    
    // Add listeners for parameter changes
    document.getElementById('motion-noise').addEventListener('input', (e) => {
        slam.motionNoise = parseInt(e.target.value) / 100;
    });
    
    document.getElementById('measurement-noise').addEventListener('input', (e) => {
        slam.measurementNoise = parseInt(e.target.value) / 100;
    });
    
    document.getElementById('num-particles').addEventListener('input', (e) => {
        slam.numParticles = parseInt(e.target.value);
        if (slam.algorithm === 'fastslam') {
            slam.initializeParticles();
        }
    });
    
    // Add listeners for visualization toggles
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', () => slam.draw());
    });
}); 