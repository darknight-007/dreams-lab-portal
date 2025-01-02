class Particle {
    constructor(x, y, theta) {
        this.x = x;
        this.y = y;
        this.theta = theta;
        this.weight = 1.0;
    }

    // Move particle according to motion model
    move(dx, dy, dtheta, noise) {
        // Add noise to raw motion
        const noisyDx = dx * (1 + noise.translation());
        const noisyDy = dy * (1 + noise.translation());
        const noisyDtheta = dtheta + noise.rotation();

        // Update position and orientation
        this.x += noisyDx;
        this.y += noisyDy;
        this.theta += noisyDtheta;
        
        // Normalize angle to [-π, π]
        this.theta = Math.atan2(Math.sin(this.theta), Math.cos(this.theta));
    }

    // Update weight based on sensor measurements with multimodal support
    updateWeight(measurements, landmarks, noise) {
        this.weight = 1.0;
        
        for (let i = 0; i < measurements.length; i++) {
            const measurement = measurements[i];
            const landmark = landmarks[i];
            
            // Calculate expected measurement
            const dx = landmark.x - this.x;
            const dy = landmark.y - this.y;
            const expectedDist = Math.sqrt(dx * dx + dy * dy);
            const expectedAngle = Math.atan2(dy, dx) - this.theta;
            
            // Normalize angle difference to [-π, π]
            const angleDiff = Math.atan2(
                Math.sin(measurement.angle - expectedAngle),
                Math.cos(measurement.angle - expectedAngle)
            );
            
            // Calculate likelihood with proper noise scaling
            const distProb = noise.gaussian(
                measurement.distance - expectedDist,
                noise.measurement * Math.max(1, expectedDist * 0.1)  // Scale with distance
            );
            const angleProb = noise.gaussian(angleDiff, noise.measurement * 0.5);
            
            // Combine probabilities with minimum threshold
            this.weight *= Math.max(0.01, distProb * angleProb);
        }
    }

    // Create a copy of the particle with adaptive random perturbation
    clone() {
        // Scale perturbation inversely with weight
        const scale = 2 * Math.sqrt(1 - this.weight);  // Non-linear scaling
        const copy = new Particle(
            this.x + (Math.random() - 0.5) * scale * 2,  // Larger position perturbation
            this.y + (Math.random() - 0.5) * scale * 2,
            this.theta + (Math.random() - 0.5) * scale * 0.2  // Smaller angle perturbation
        );
        copy.weight = this.weight;
        return copy;
    }
}

class ParticleFilter {
    constructor(numParticles, bounds, noise) {
        this.particles = [];
        this.noise = noise;
        this.bounds = bounds;
        
        // Initialize particles in a grid pattern with small random offsets
        const cols = Math.ceil(Math.sqrt(numParticles));
        const rows = Math.ceil(numParticles / cols);
        const dx = (bounds.xMax - bounds.xMin) / cols;
        const dy = (bounds.yMax - bounds.yMin) / rows;
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                if (this.particles.length < numParticles) {
                    const x = bounds.xMin + (j + 0.5 + (Math.random() - 0.5) * 0.2) * dx;
                    const y = bounds.yMin + (i + 0.5 + (Math.random() - 0.5) * 0.2) * dy;
                    const theta = Math.random() * 2 * Math.PI - Math.PI;
                    this.particles.push(new Particle(x, y, theta));
                }
            }
        }
    }

    // Move all particles according to motion model
    predict(dx, dy, dtheta) {
        this.particles.forEach(particle => {
            particle.move(dx, dy, dtheta, this.noise);
        });
    }

    // Update particle weights based on measurements
    update(measurements, landmarks) {
        // Update weights
        this.particles.forEach(particle => {
            particle.updateWeight(measurements, landmarks, this.noise);
        });

        // Normalize weights with adaptive minimum weight threshold
        const totalWeight = this.particles.reduce((sum, p) => sum + p.weight, 0);
        if (totalWeight > 0) {
            const avgWeight = totalWeight / this.particles.length;
            const minWeight = avgWeight * 0.1; // Adaptive minimum weight
            this.particles.forEach(p => {
                p.weight = Math.max(minWeight, p.weight / totalWeight);
            });
            
            // Renormalize after applying minimum weight
            const newTotal = this.particles.reduce((sum, p) => sum + p.weight, 0);
            this.particles.forEach(p => p.weight /= newTotal);
        } else {
            // If all weights are zero, reinitialize with uniform weights
            const w = 1.0 / this.particles.length;
            this.particles.forEach(p => p.weight = w);
        }
    }

    // Resample particles based on their weights with added diversity
    resample() {
        const newParticles = [];
        
        // Compute cumulative weights
        const cumWeights = [];
        let sum = 0;
        this.particles.forEach(p => {
            sum += p.weight;
            cumWeights.push(sum);
        });

        // Low variance resampling
        const step = 1.0 / this.particles.length;
        let u = Math.random() * step;
        let j = 0;
        
        for (let i = 0; i < this.particles.length; i++) {
            while (u > cumWeights[j]) {
                j++;
            }
            // Clone with small random perturbation to maintain diversity
            newParticles.push(this.particles[j].clone());
            u += step;
        }

        this.particles = newParticles;
    }

    // Get estimated pose (weighted mean of particles)
    getEstimatedPose() {
        let x = 0, y = 0;
        let cosSum = 0, sinSum = 0;
        
        this.particles.forEach(p => {
            x += p.weight * p.x;
            y += p.weight * p.y;
            cosSum += p.weight * Math.cos(p.theta);
            sinSum += p.weight * Math.sin(p.theta);
        });

        return {
            x: x,
            y: y,
            theta: Math.atan2(sinSum, cosSum)
        };
    }

    // Get particle positions and weights for visualization
    getParticleData() {
        return this.particles.map(p => ({
            x: p.x,
            y: p.y,
            theta: p.theta,
            weight: p.weight
        }));
    }
}

// Noise models and utilities
const NoiseModels = {
    // Gaussian noise
    gaussian: (x, std) => {
        const variance = std * std;
        return Math.exp(-x * x / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);
    },

    // Motion noise model
    createMotionNoise: (translationNoise, rotationNoise) => ({
        translation: () => (Math.random() * 2 - 1) * translationNoise,
        rotation: () => (Math.random() * 2 - 1) * rotationNoise
    }),

    // Measurement noise model
    createMeasurementNoise: (std) => ({
        measurement: std,
        gaussian: (x, std) => NoiseModels.gaussian(x, std)
    })
}; 