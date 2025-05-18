class GaussianProcess {
    constructor(kernel, noiseLevel) {
        this.kernel = kernel;
        this.noiseLevel = noiseLevel;
        this.X = null;
        this.y = null;
        this.L = null;
        this.alpha = null;
    }
    
    fit(X, y) {
        this.X = X;
        this.y = y;
        
        // Compute kernel matrix
        const n = X.length;
        const K = Array(n).fill().map(() => Array(n).fill(0));
        
        for(let i = 0; i < n; i++) {
            for(let j = 0; j <= i; j++) {
                K[i][j] = this.kernel(X[i], X[j]);
                if(i === j) K[i][j] += this.noiseLevel;
                K[j][i] = K[i][j];
            }
        }
        
        // Cholesky decomposition
        this.L = this.choleskyDecomposition(K);
        
        // Solve for alpha
        this.alpha = this.choleskySolve(this.L, y);
    }
    
    predict(x) {
        if(!this.X || !this.y) return { mean: 0, variance: this.kernel(x, x) };
        
        // Compute k*
        const kstar = this.X.map(xi => this.kernel(x, xi));
        
        // Compute mean
        const mean = kstar.reduce((sum, ki, i) => sum + ki * this.alpha[i], 0);
        
        // Compute variance
        const v = this.choleskySolve(this.L, kstar);
        const variance = this.kernel(x, x) - v.reduce((sum, vi, i) => sum + vi * kstar[i], 0);
        
        return { mean, variance: Math.max(variance, 0) };
    }
    
    choleskyDecomposition(A) {
        const n = A.length;
        const L = Array(n).fill().map(() => Array(n).fill(0));
        
        for(let i = 0; i < n; i++) {
            for(let j = 0; j <= i; j++) {
                let sum = 0;
                
                if(j === i) {
                    for(let k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[i][j] = Math.sqrt(A[i][i] - sum);
                } else {
                    for(let k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        
        return L;
    }
    
    choleskySolve(L, b) {
        const n = L.length;
        const y = Array(n).fill(0);
        const x = Array(n).fill(0);
        
        // Forward substitution
        for(let i = 0; i < n; i++) {
            let sum = 0;
            for(let j = 0; j < i; j++) {
                sum += L[i][j] * y[j];
            }
            y[i] = (b[i] - sum) / L[i][i];
        }
        
        // Backward substitution
        for(let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for(let j = i + 1; j < n; j++) {
                sum += L[j][i] * x[j];
            }
            x[i] = (y[i] - sum) / L[i][i];
        }
        
        return x;
    }
    
    samplePrior(x, numSamples = 1) {
        const n = x.length;
        const K = Array(n).fill().map(() => Array(n).fill(0));
        
        // Compute kernel matrix
        for(let i = 0; i < n; i++) {
            for(let j = 0; j <= i; j++) {
                K[i][j] = this.kernel(x[i], x[j]);
                K[j][i] = K[i][j];
            }
        }
        
        // Cholesky decomposition
        const L = this.choleskyDecomposition(K);
        
        // Generate samples
        const samples = Array(numSamples).fill().map(() => {
            const z = Array(n).fill().map(() => this.randn());
            const f = Array(n).fill(0);
            
            // f = L * z
            for(let i = 0; i < n; i++) {
                for(let j = 0; j <= i; j++) {
                    f[i] += L[i][j] * z[j];
                }
            }
            
            return f;
        });
        
        return samples;
    }
    
    samplePosterior(x, numSamples = 1) {
        const predictions = x.map(xi => this.predict(xi));
        const means = predictions.map(p => p.mean);
        const variances = predictions.map(p => p.variance);
        
        // Generate samples
        const samples = Array(numSamples).fill().map(() => {
            return means.map((mu, i) => mu + Math.sqrt(variances[i]) * this.randn());
        });
        
        return samples;
    }
    
    randn() {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

// Kernel functions
GaussianProcess.RBF = (lengthScale) => {
    return (x1, x2) => Math.exp(-0.5 * Math.pow(x1 - x2, 2) / (lengthScale * lengthScale));
};

GaussianProcess.Periodic = (lengthScale, period) => {
    return (x1, x2) => Math.exp(-2 * Math.pow(Math.sin(Math.PI * Math.abs(x1 - x2) / period), 2) / (lengthScale * lengthScale));
};

GaussianProcess.Matern32 = (lengthScale) => {
    return (x1, x2) => {
        const d = Math.abs(x1 - x2) / lengthScale;
        const sqrt3 = Math.sqrt(3);
        return (1 + sqrt3 * d) * Math.exp(-sqrt3 * d);
    };
}; 