class KalmanFilter {
    constructor() {
        const dt = 0.1;  // Time step
        
        // State vector [x, y, vx, vy]
        this.state = [0, 0, 0, 0];
        
        // State covariance matrix (initialized with high uncertainty)
        this.P = [
            [100, 0, 0, 0],
            [0, 100, 0, 0],
            [0, 0, 10, 0],  // Lower initial uncertainty for velocity
            [0, 0, 0, 10]
        ];
        
        // State transition matrix (constant velocity model)
        this.F = [
            [1, 0, dt, 0],   // x = x + vx*dt
            [0, 1, 0, dt],   // y = y + vy*dt
            [0, 0, 1, 0],    // vx = vx
            [0, 0, 0, 1]     // vy = vy
        ];
        
        // Control input matrix (for IMU velocity measurements)
        this.B = [
            [dt*dt/2, 0],    // x acceleration influence
            [0, dt*dt/2],    // y acceleration influence
            [dt, 0],         // vx direct influence
            [0, dt]          // vy direct influence
        ];
        
        // Measurement matrix (GPS measures position only)
        this.H = [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ];

        // Default process noise (Q) and measurement noise (R) matrices
        this.defaultQ = [
            [dt*dt*dt*dt/4, 0, dt*dt*dt/2, 0],
            [0, dt*dt*dt*dt/4, 0, dt*dt*dt/2],
            [dt*dt*dt/2, 0, dt*dt, 0],
            [0, dt*dt*dt/2, 0, dt*dt]
        ].map(row => row.map(val => val * 0.1));  // Scale by base noise

        this.defaultR = [
            [1, 0],
            [0, 1]
        ];
    }

    // Matrix multiplication with dimension checking
    matMul(a, b) {
        if (a[0].length !== b.length) {
            throw new Error('Matrix dimensions do not match for multiplication');
        }
        const result = Array(a.length).fill().map(() => Array(b[0].length).fill(0));
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b[0].length; j++) {
                for (let k = 0; k < b.length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    // Matrix transpose
    transpose(matrix) {
        return matrix[0].map((_, i) => matrix.map(row => row[i]));
    }

    // Matrix addition with dimension checking
    matAdd(a, b) {
        if (a.length !== b.length || a[0].length !== b[0].length) {
            throw new Error('Matrix dimensions do not match for addition');
        }
        return a.map((row, i) => row.map((val, j) => val + b[i][j]));
    }

    // Matrix subtraction with dimension checking
    matSub(a, b) {
        if (a.length !== b.length || a[0].length !== b[0].length) {
            throw new Error('Matrix dimensions do not match for subtraction');
        }
        return a.map((row, i) => row.map((val, j) => val - b[i][j]));
    }

    // Matrix inverse (2x2 only) with determinant check
    inverse2x2(matrix) {
        const det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        if (Math.abs(det) < 1e-10) {
            throw new Error('Matrix is singular or nearly singular');
        }
        return [
            [matrix[1][1] / det, -matrix[0][1] / det],
            [-matrix[1][0] / det, matrix[0][0] / det]
        ];
    }

    // Predict step with optional control input
    predict(Q = this.defaultQ, u = null) {
        // State prediction: x = Fx + Bu
        const Fx = this.matMul(this.F, [[this.state[0]], [this.state[1]], [this.state[2]], [this.state[3]]]);
        
        // Add control input if provided (IMU velocities)
        if (u && u.length === 4) {
            const Bu = this.matMul(this.B, [[u[2]], [u[3]]]);  // Use velocity components
            this.state = Fx.map((val, i) => val[0] + Bu[i][0]);
        } else {
            this.state = Fx.map(row => row[0]);
        }

        // Covariance prediction: P = FPF' + Q
        const FP = this.matMul(this.F, this.P);
        const FT = this.transpose(this.F);
        this.P = this.matAdd(this.matMul(FP, FT), Q);

        // Ensure symmetry and positive definiteness of covariance matrix
        this.P = this.P.map((row, i) => row.map((val, j) => {
            if (i === j) return Math.max(val, 1e-10);  // Ensure positive diagonal
            const symVal = (val + this.P[j][i]) / 2;   // Symmetrize
            return Math.abs(symVal) < 1e-10 ? 0 : symVal;  // Clean small values
        }));
    }

    // Update step with innovation gating and adaptive noise
    update(measurement, R = this.defaultR) {
        // Innovation (measurement residual): y = z - Hx
        const predicted_measurement = this.matMul(this.H, [[this.state[0]], [this.state[1]], [this.state[2]], [this.state[3]]]);
        const y = this.matSub(
            [[measurement[0]], [measurement[1]]],
            predicted_measurement
        );

        // Innovation covariance: S = HPH' + R
        const HP = this.matMul(this.H, this.P);
        const S = this.matAdd(
            this.matMul(HP, this.transpose(this.H)),
            R
        );

        try {
            // Kalman gain: K = PH'S^(-1)
            const K = this.matMul(
                this.matMul(this.P, this.transpose(this.H)),
                this.inverse2x2(S)
            );

            // Innovation gating (Mahalanobis distance)
            const yT = this.transpose(y);
            const Sinv = this.inverse2x2(S);
            const mahalanobis = Math.sqrt(this.matMul(this.matMul(yT, Sinv), y)[0][0]);
            
            // Only update if innovation is within acceptable bounds (3-sigma gate)
            if (mahalanobis < 3.0) {
                // State update: x = x + Ky
                const stateUpdate = this.matMul(K, y).map(row => row[0]);
                this.state = this.state.map((val, i) => val + stateUpdate[i]);

                // Joseph form covariance update: P = (I-KH)P(I-KH)' + KRK'
                // This form guarantees positive definiteness
                const I = [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ];
                const KH = this.matMul(K, this.H);
                const IKH = this.matSub(I, KH);
                const IKHP = this.matMul(IKH, this.P);
                const IKHPIKHP = this.matMul(IKHP, this.transpose(IKH));
                const KRKt = this.matMul(this.matMul(K, R), this.transpose(K));
                this.P = this.matAdd(IKHPIKHP, KRKt);

                // Ensure symmetry and positive definiteness
                this.P = this.P.map((row, i) => row.map((val, j) => {
                    if (i === j) return Math.max(val, 1e-10);
                    const symVal = (val + this.P[j][i]) / 2;
                    return Math.abs(symVal) < 1e-10 ? 0 : symVal;
                }));
            }
        } catch (error) {
            console.error('Matrix operation error:', error);
            // Keep previous state and covariance if update fails
        }
    }

    // Get current state and uncertainty
    getState() {
        return {
            position: [this.state[0], this.state[1]],
            velocity: [this.state[2], this.state[3]],
            uncertainty: {
                position: [Math.sqrt(this.P[0][0]), Math.sqrt(this.P[1][1])],
                velocity: [Math.sqrt(this.P[2][2]), Math.sqrt(this.P[3][3])]
            }
        };
    }
} 