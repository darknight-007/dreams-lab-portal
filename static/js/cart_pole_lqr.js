class CartPoleLQR {
    constructor() {
        this.canvas = document.getElementById('cartPoleCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // System parameters
        this.params = {
            cartMass: 1.0,      // kg
            poleMass: 0.1,      // kg
            poleLength: 1.0,    // m
            gravity: 9.81,      // m/s^2
            friction: 0.1,      // damping coefficient
        };

        // State vector: [x, theta, x_dot, theta_dot]
        this.state = [0, 0.1, 0, 0];
        
        // LQR cost matrices
        this.Q = math.diag([1, 10, 1, 1]);  // State cost
        this.R = math.matrix([[1]]);        // Control cost

        // Animation properties
        this.isRunning = false;
        this.lastTime = 0;
        this.scale = 100;  // pixels per meter
        this.simulationTime = 0;
        this.maxSimulationTime = 30; // 30 seconds max simulation time

        // Performance metrics
        this.metrics = {
            maxAngle: 0,
            maxPosition: 0,
            totalControlEffort: 0,
            samples: 0
        };

        // Bind event handlers
        this.setupEventListeners();
        
        // Initial control gains calculation
        this.updateLQRGains();
        
        // Start animation loop
        this.animate();
    }

    setupEventListeners() {
        // System parameter controls
        document.getElementById('cartMass').addEventListener('input', (e) => {
            this.params.cartMass = parseFloat(e.target.value);
            document.getElementById('cartMassValue').textContent = e.target.value;
            this.updateLQRGains();
        });

        document.getElementById('poleMass').addEventListener('input', (e) => {
            this.params.poleMass = parseFloat(e.target.value);
            document.getElementById('poleMassValue').textContent = e.target.value;
            this.updateLQRGains();
        });

        document.getElementById('poleLength').addEventListener('input', (e) => {
            this.params.poleLength = parseFloat(e.target.value);
            document.getElementById('poleLengthValue').textContent = e.target.value;
            this.updateLQRGains();
        });

        // LQR cost parameter controls
        document.getElementById('positionCost').addEventListener('input', (e) => {
            this.Q._data[0][0] = parseFloat(e.target.value);
            document.getElementById('positionCostValue').textContent = e.target.value;
            this.updateLQRGains();
        });

        document.getElementById('angleCost').addEventListener('input', (e) => {
            this.Q._data[1][1] = parseFloat(e.target.value);
            document.getElementById('angleCostValue').textContent = e.target.value;
            this.updateLQRGains();
        });

        document.getElementById('controlCost').addEventListener('input', (e) => {
            this.R._data[0][0] = parseFloat(e.target.value);
            document.getElementById('controlCostValue').textContent = e.target.value;
            this.updateLQRGains();
        });

        // Control buttons
        document.getElementById('startBtn').addEventListener('click', () => {
            if (!this.isRunning) {
                this.startSimulation();
            } else {
                this.pauseSimulation();
            }
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetSimulation();
        });

        document.getElementById('perturbBtn').addEventListener('click', () => {
            if (this.isRunning) {
                this.state[1] += 0.1; // Add perturbation to angle
                this.metrics.maxAngle = Math.max(this.metrics.maxAngle, Math.abs(this.state[1]));
            }
        });
    }

    startSimulation() {
        this.isRunning = true;
        this.simulationTime = 0;
        this.lastTime = 0;
        this.metrics = {
            maxAngle: Math.abs(this.state[1]),
            maxPosition: Math.abs(this.state[0]),
            totalControlEffort: 0,
            samples: 0
        };
        document.getElementById('startBtn').textContent = 'Pause';
    }

    pauseSimulation() {
        this.isRunning = false;
        document.getElementById('startBtn').textContent = 'Start';
    }

    resetSimulation() {
        this.pauseSimulation();
        this.state = [0, 0.1, 0, 0];
        this.simulationTime = 0;
        this.metrics = {
            maxAngle: Math.abs(this.state[1]),
            maxPosition: Math.abs(this.state[0]),
            totalControlEffort: 0,
            samples: 0
        };
        this.draw();
    }

    checkSimulationCompletion() {
        const stateThreshold = 0.01;
        const velocityThreshold = 0.01;
        const timeThreshold = 5.0; // Minimum simulation time

        // Check if system has stabilized
        const isStable = Math.abs(this.state[0]) < stateThreshold && 
                        Math.abs(this.state[1]) < stateThreshold &&
                        Math.abs(this.state[2]) < velocityThreshold &&
                        Math.abs(this.state[3]) < velocityThreshold;

        // Check if simulation has run for minimum time
        const hasMinTime = this.simulationTime >= timeThreshold;

        // Check if simulation has exceeded maximum time
        const hasMaxTime = this.simulationTime >= this.maxSimulationTime;

        // Check if system has failed (pole fallen or cart too far)
        const hasFailed = Math.abs(this.state[1]) > Math.PI/2 || Math.abs(this.state[0]) > 2.0;

        if ((isStable && hasMinTime) || hasMaxTime || hasFailed) {
            this.pauseSimulation();
            this.displayPerformanceMetrics();
        }
    }

    displayPerformanceMetrics() {
        const avgControlEffort = this.metrics.totalControlEffort / this.metrics.samples;
        console.log('Simulation Complete:');
        console.log(`Max Angle Deviation: ${this.metrics.maxAngle.toFixed(3)} rad`);
        console.log(`Max Position Deviation: ${this.metrics.maxPosition.toFixed(3)} m`);
        console.log(`Average Control Effort: ${avgControlEffort.toFixed(3)} N`);
        console.log(`Simulation Time: ${this.simulationTime.toFixed(2)} s`);
    }

    getLinearizedSystem() {
        const { cartMass: M, poleMass: m, poleLength: l, gravity: g } = this.params;
        
        // Linearized system matrices at the upright equilibrium
        const A = math.matrix([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, m*g/M, 0, 0],
            [0, (M+m)*g/(M*l), 0, 0]
        ]);

        const B = math.matrix([
            [0],
            [0],
            [1/M],
            [1/(M*l)]
        ]);

        return { A, B };
    }

    updateLQRGains() {
        const { A, B } = this.getLinearizedSystem();
        this.K = this.solveLQR(A, B, this.Q, this.R);
    }

    solveLQR(A, B, Q, R) {
        // Solve Algebraic Riccati Equation using iterative method
        let P = math.identity(4);
        const maxIter = 1000;
        const tolerance = 1e-6;

        for (let i = 0; i < maxIter; i++) {
            const P_next = math.add(
                Q,
                math.add(
                    math.multiply(
                        math.transpose(A),
                        math.multiply(P, A)
                    ),
                    math.multiply(
                        math.multiply(
                            math.multiply(
                                math.transpose(A),
                                P
                            ),
                            B
                        ),
                        math.multiply(
                            math.inv(
                                math.add(
                                    R,
                                    math.multiply(
                                        math.multiply(
                                            math.transpose(B),
                                            P
                                        ),
                                        B
                                    )
                                )
                            ),
                            math.multiply(
                                math.multiply(
                                    math.transpose(B),
                                    P
                                ),
                                A
                            )
                        )
                    )
                )
            );

            if (math.norm(math.subtract(P_next, P)) < tolerance) {
                P = P_next;
                break;
            }
            P = P_next;
        }

        // Calculate optimal feedback gain K
        const K = math.multiply(
            math.inv(R),
            math.multiply(
                math.multiply(
                    math.transpose(B),
                    P
                ),
                A
            )
        );

        return K;
    }

    computeControl() {
        // Compute control input u = -Kx
        const stateVector = math.matrix([this.state]);
        const u = math.multiply(math.multiply(-1, this.K), math.transpose(stateVector));
        return u._data[0][0];
    }

    updateState(dt) {
        const { cartMass: M, poleMass: m, poleLength: l, gravity: g, friction: b } = this.params;
        
        // Get control input
        const u = this.computeControl();
        
        // Current state
        const [x, theta, x_dot, theta_dot] = this.state;
        
        // Update metrics
        this.metrics.maxAngle = Math.max(this.metrics.maxAngle, Math.abs(theta));
        this.metrics.maxPosition = Math.max(this.metrics.maxPosition, Math.abs(x));
        this.metrics.totalControlEffort += Math.abs(u);
        this.metrics.samples++;
        
        // Compute accelerations using full nonlinear dynamics
        const sin_theta = Math.sin(theta);
        const cos_theta = Math.cos(theta);
        
        const num1 = u + m*l*theta_dot*theta_dot*sin_theta - b*x_dot;
        const num2 = g*sin_theta - (num1*cos_theta)/(M + m);
        const den = l*(4/3 - (m*cos_theta*cos_theta)/(M + m));
        
        const theta_ddot = num2/den;
        const x_ddot = (num1 - m*l*theta_ddot*cos_theta)/(M + m);
        
        // Euler integration
        this.state[0] = x + x_dot*dt;
        this.state[1] = theta + theta_dot*dt;
        this.state[2] = x_dot + x_ddot*dt;
        this.state[3] = theta_dot + theta_ddot*dt;

        // Update simulation time
        this.simulationTime += dt;

        // Update display values
        document.getElementById('positionValue').textContent = x.toFixed(2);
        document.getElementById('angleValue').textContent = theta.toFixed(2);
        document.getElementById('controlValue').textContent = u.toFixed(2);

        // Check if simulation should complete
        this.checkSimulationCompletion();
    }

    draw() {
        const { ctx, canvas, scale } = this;
        const [x, theta] = this.state;
        const { poleLength: l } = this.params;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Set origin to center-bottom of canvas
        const originX = canvas.width/2;
        const originY = canvas.height - 50;

        // Draw track
        ctx.beginPath();
        ctx.moveTo(0, originY);
        ctx.lineTo(canvas.width, originY);
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw cart
        const cartWidth = 50;
        const cartHeight = 30;
        const cartX = originX + x*scale - cartWidth/2;
        const cartY = originY - cartHeight;
        
        ctx.fillStyle = '#4CAF50';
        ctx.fillRect(cartX, cartY, cartWidth, cartHeight);

        // Draw pole
        const poleX = originX + x*scale;
        const poleY = cartY;
        const endX = poleX + l*scale*Math.sin(theta);
        const endY = poleY - l*scale*Math.cos(theta);

        ctx.beginPath();
        ctx.moveTo(poleX, poleY);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = '#2196F3';
        ctx.lineWidth = 6;
        ctx.stroke();

        // Draw bob at end of pole
        ctx.beginPath();
        ctx.arc(endX, endY, 10, 0, 2*Math.PI);
        ctx.fillStyle = '#2196F3';
        ctx.fill();
    }

    animate(currentTime = 0) {
        if (this.lastTime === 0) this.lastTime = currentTime;
        const dt = Math.min((currentTime - this.lastTime) / 1000, 0.05); // Cap at 20 Hz
        this.lastTime = currentTime;

        if (this.isRunning) {
            this.updateState(dt);
        }

        this.draw();
        requestAnimationFrame((time) => this.animate(time));
    }
}

// Initialize when the page loads
window.addEventListener('load', () => {
    new CartPoleLQR();
}); 