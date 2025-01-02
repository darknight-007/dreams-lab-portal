class PathPlanningBuddy {
    constructor() {
        // Initialize state first
        this.start = null;
        this.goal = null;
        this.obstacles = [];
        this.path = [];
        this.tree = [];
        this.samples = [];
        this.isRunning = false;
        this.costs = new Map(); // Store costs from start to each node
        this.comparisonPaths = new Map(); // Initialize comparison paths
        this.comparisonStats = new Map(); // Initialize comparison stats
        
        // Algorithm colors
        this.algorithmColors = {
            'rrt': '#4CAF50',
            'rrtstar': '#2196F3',
            'informed': '#9C27B0',
            'prm': '#FF9800'
        };
        
        // Initialize parameters
        this.stepSize = 20;
        this.goalBias = 0.2;
        this.maxIterations = 5000;
        this.searchRadius = 50; // For RRT*
        this.algorithm = 'rrt'; // Default algorithm
        
        // Then setup canvas
        this.canvas = document.getElementById('path-planning-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();
        
        // Bind event listeners
        window.addEventListener('resize', () => this.resizeCanvas());
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Add algorithm change listener
        document.getElementById('algorithm').addEventListener('change', (e) => {
            this.algorithm = e.target.value;
            // Show/hide relevant parameters
            document.querySelectorAll('.rrtstar-only').forEach(el => {
                el.style.display = (this.algorithm === 'rrt') ? 'none' : 'block';
            });
        });
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.draw();
    }

    reset() {
        this.start = null;
        this.goal = null;
        this.obstacles = [];
        this.path = [];
        this.tree = [];
        this.samples = [];
        this.costs = new Map();
        this.isRunning = false;
        this.draw();
    }

    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Only handle left clicks
        if (e.button === 0) {
            // Shift + Left click: Set start point
            if (e.shiftKey) {
                this.start = { x, y };
            }
            // Ctrl + Left click: Set goal point 
            else if (e.ctrlKey) {
                this.goal = { x, y };
            }
            // Left click only: Add obstacle
            else {
                this.obstacles.push({ x, y, radius: 20 });
            }
        }
        
        this.draw();
    }

    // RRT/RRT* Implementation
    rrtStep() {
        if (this.tree.length === 0 && this.start) {
            this.tree.push([null, this.start]);
            this.costs.set(this.start, 0); // Initialize cost for start node
        }

        // Sample point with goal bias
        let sample;
        if (Math.random() < this.goalBias && this.goal) {
            sample = { x: this.goal.x, y: this.goal.y };
        } else {
            sample = {
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height
            };
        }
        this.samples.push(sample);

        // Find nearest node
        const nearest = this.findNearestNode(sample);
        if (!nearest) return false;

        // Extend tree
        const newNode = this.extend(nearest, sample);
        if (!newNode || !this.isValidPath(nearest, newNode)) return false;

        if (this.algorithm === 'rrt') {
            // Standard RRT
            this.tree.push([nearest, newNode]);
            this.costs.set(newNode, this.costs.get(nearest) + this.distanceBetween(nearest, newNode));
        } else {
            // RRT* with rewiring
            const nearNodes = this.findNodesInRadius(newNode, this.searchRadius);
            let minParent = nearest;
            let minCost = this.costs.get(nearest) + this.distanceBetween(nearest, newNode);

            // Find best parent
            for (const node of nearNodes) {
                if (node === nearest) continue;
                const potentialCost = this.costs.get(node) + this.distanceBetween(node, newNode);
                if (potentialCost < minCost && this.isValidPath(node, newNode)) {
                    minParent = node;
                    minCost = potentialCost;
                }
            }

            // Add new node with best parent
            this.tree.push([minParent, newNode]);
            this.costs.set(newNode, minCost);

            // Rewire tree
            for (const node of nearNodes) {
                if (node === minParent || node === newNode) continue;
                const potentialCost = minCost + this.distanceBetween(newNode, node);
                if (potentialCost < this.costs.get(node) && this.isValidPath(newNode, node)) {
                    // Remove old edge
                    this.tree = this.tree.filter(([from, to]) => to !== node);
                    // Add new edge
                    this.tree.push([newNode, node]);
                    // Update cost
                    this.costs.set(node, potentialCost);
                }
            }
        }

        // Check if we reached the goal
        if (this.goal && this.distanceBetween(newNode, this.goal) < this.stepSize) {
            this.tree.push([newNode, this.goal]);
            this.costs.set(this.goal, this.costs.get(newNode) + this.distanceBetween(newNode, this.goal));
            return true;
        }

        return false;
    }

    findNearestNode(point) {
        let nearest = null;
        let minDist = Infinity;

        for (const [_, node] of this.tree) {
            if (!node) continue;
            const dist = this.distanceBetween(point, node);
            if (dist < minDist) {
                nearest = node;
                minDist = dist;
            }
        }

        return nearest;
    }

    extend(from, to) {
        const dist = this.distanceBetween(from, to);
        if (dist < this.stepSize) return to;

        const theta = Math.atan2(to.y - from.y, to.x - from.x);
        return {
            x: from.x + this.stepSize * Math.cos(theta),
            y: from.y + this.stepSize * Math.sin(theta)
        };
    }

    isValidPath(from, to) {
        for (const obstacle of this.obstacles) {
            // Check if line intersects with obstacle
            const d = this.pointToLineDistance(obstacle, from, to);
            if (d < obstacle.radius) return false;
        }
        return true;
    }

    pointToLineDistance(point, lineStart, lineEnd) {
        const numerator = Math.abs(
            (lineEnd.y - lineStart.y) * point.x -
            (lineEnd.x - lineStart.x) * point.y +
            lineEnd.x * lineStart.y -
            lineEnd.y * lineStart.x
        );
        const denominator = Math.sqrt(
            Math.pow(lineEnd.y - lineStart.y, 2) +
            Math.pow(lineEnd.x - lineStart.x, 2)
        );
        return numerator / denominator;
    }

    distanceBetween(p1, p2) {
        return Math.sqrt(
            Math.pow(p2.x - p1.x, 2) +
            Math.pow(p2.y - p1.y, 2)
        );
    }

    reconstructPath() {
        if (!this.goal) return;
        
        this.path = [this.goal];
        let current = this.goal;
        
        while (current) {
            const edge = this.tree.find(([from, to]) => to === current);
            if (!edge) break;
            current = edge[0];
            if (current) this.path.unshift(current);
        }
    }

    draw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw obstacles
        this.ctx.fillStyle = '#666';
        for (const obstacle of this.obstacles) {
            this.ctx.beginPath();
            this.ctx.arc(obstacle.x, obstacle.y, obstacle.radius, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // Draw tree if enabled
        if (document.getElementById('show-tree')?.checked && this.tree.length > 0) {
            this.ctx.strokeStyle = '#aaa';
            this.ctx.lineWidth = 1;
            for (const [from, to] of this.tree) {
                if (!from || !to) continue;
                this.ctx.beginPath();
                this.ctx.moveTo(from.x, from.y);
                this.ctx.lineTo(to.x, to.y);
                this.ctx.stroke();
            }
        }
        
        // Draw samples if enabled
        if (document.getElementById('show-samples')?.checked && this.samples.length > 0) {
            this.ctx.fillStyle = '#999';
            for (const sample of this.samples) {
                this.ctx.beginPath();
                this.ctx.arc(sample.x, sample.y, 2, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
        
        // Draw comparison paths if they exist
        for (const [algo, path] of this.comparisonPaths.entries()) {
            if (path.length > 0) {
                this.ctx.strokeStyle = this.algorithmColors[algo];
                this.ctx.lineWidth = 3;
                this.ctx.beginPath();
                this.ctx.moveTo(path[0].x, path[0].y);
                for (let i = 1; i < path.length; i++) {
                    this.ctx.lineTo(path[i].x, path[i].y);
                }
                this.ctx.stroke();
            }
        }
        
        // Draw current path if not in comparison mode
        if (this.comparisonPaths.size === 0 && this.path.length > 0) {
            this.ctx.strokeStyle = this.algorithmColors[this.algorithm];
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(this.path[0].x, this.path[0].y);
            for (let i = 1; i < this.path.length; i++) {
                this.ctx.lineTo(this.path[i].x, this.path[i].y);
            }
            this.ctx.stroke();
        }
        
        // Draw start point
        if (this.start) {
            this.ctx.fillStyle = '#4CAF50';
            this.ctx.beginPath();
            this.ctx.arc(this.start.x, this.start.y, 8, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // Draw goal point
        if (this.goal) {
            this.ctx.fillStyle = '#f44336';
            this.ctx.beginPath();
            this.ctx.arc(this.goal.x, this.goal.y, 8, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    clearAll() {
        this.start = null;
        this.goal = null;
        this.obstacles = [];
        this.path = [];
        this.tree = [];
        this.samples = [];
        this.costs = new Map();
        this.isRunning = false;
        this.draw();
    }

    generateRandomScene() {
        // Clear everything first
        this.clearAll();

        // Set random start and goal points
        const margin = 50;
        this.start = {
            x: margin + Math.random() * (this.canvas.width - 2 * margin),
            y: margin + Math.random() * (this.canvas.height - 2 * margin)
        };
        this.goal = {
            x: margin + Math.random() * (this.canvas.width - 2 * margin),
            y: margin + Math.random() * (this.canvas.height - 2 * margin)
        };

        // Add random obstacles
        const numObstacles = 10 + Math.floor(Math.random() * 10);
        for (let i = 0; i < numObstacles; i++) {
            const obstacle = {
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: 15 + Math.random() * 15
            };
            // Don't add if too close to start or goal
            if (this.distanceBetween(obstacle, this.start) > obstacle.radius * 2 &&
                this.distanceBetween(obstacle, this.goal) > obstacle.radius * 2) {
                this.obstacles.push(obstacle);
            }
        }
        this.draw();
    }

    // Start planning
    async startPlanning() {
        if (!this.start || !this.goal) {
            alert('Please set start and goal points first');
            return;
        }

        // Reset planning state
        this.isRunning = true;
        this.path = [];
        this.tree = [];
        this.samples = [];
        this.costs = new Map();

        // Get parameters from UI
        this.stepSize = parseInt(document.getElementById('step-size').value);
        this.goalBias = parseInt(document.getElementById('goal-bias').value) / 100;
        this.maxIterations = parseInt(document.getElementById('max-iterations').value);
        this.searchRadius = parseInt(document.getElementById('search-radius').value);
        this.algorithm = document.getElementById('algorithm').value;

        // Start RRT/RRT*
        let iterations = 0;
        const startTime = performance.now();
        let success = false;

        while (this.isRunning && iterations < this.maxIterations) {
            if (this.rrtStep()) {
                success = true;
                this.reconstructPath();
                const endTime = performance.now();
                
                // Update statistics
                document.getElementById('path-length').textContent = 
                    `Path Length: ${this.calculatePathLength().toFixed(2)}`;
                document.getElementById('iterations').textContent = 
                    `Iterations: ${iterations}`;
                document.getElementById('computation-time').textContent = 
                    `Time: ${(endTime - startTime).toFixed(2)} ms`;
                document.getElementById('success-rate').textContent = 
                    `Success Rate: 100%`;
                
                break;
            }
            iterations++;
            
            // Update visualization periodically
            if (iterations % 10 === 0) {
                this.draw();
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        if (!success) {
            document.getElementById('success-rate').textContent = 'Success Rate: 0%';
        }

        this.isRunning = false;
        this.draw();
    }

    calculatePathLength() {
        let length = 0;
        for (let i = 1; i < this.path.length; i++) {
            length += this.distanceBetween(this.path[i-1], this.path[i]);
        }
        return length;
    }

    findNodesInRadius(point, radius) {
        return this.tree
            .map(([_, node]) => node)
            .filter(node => node && this.distanceBetween(point, node) <= radius);
    }

    async compareAllAlgorithms() {
        if (!this.start || !this.goal) {
            alert('Please set start and goal points first');
            return;
        }

        // Clear previous comparison results
        this.comparisonPaths.clear();
        this.comparisonStats.clear();

        // Get parameters from UI
        const stepSize = parseInt(document.getElementById('step-size').value);
        const goalBias = parseInt(document.getElementById('goal-bias').value) / 100;
        const maxIterations = parseInt(document.getElementById('max-iterations').value);
        const searchRadius = parseInt(document.getElementById('search-radius').value);

        // Create comparison table in statistics section
        const statsDiv = document.getElementById('statistics');
        statsDiv.innerHTML = `
            <h3>Comparison Results</h3>
            <table class="comparison-table">
                <tr>
                    <th>Algorithm</th>
                    <th>Path Length</th>
                    <th>Time (ms)</th>
                    <th>Iterations</th>
                </tr>
            </table>
        `;
        const table = statsDiv.querySelector('.comparison-table');

        // Run each algorithm
        const algorithms = ['rrt', 'rrtstar', 'informed', 'prm'];
        for (const algo of algorithms) {
            // Reset state
            this.path = [];
            this.tree = [];
            this.samples = [];
            this.costs = new Map();
            this.algorithm = algo;

            // Set parameters
            this.stepSize = stepSize;
            this.goalBias = goalBias;
            this.maxIterations = maxIterations;
            this.searchRadius = searchRadius;

            // Run algorithm
            const startTime = performance.now();
            let iterations = 0;
            let success = false;

            while (iterations < maxIterations) {
                if (this.rrtStep()) {
                    success = true;
                    this.reconstructPath();
                    const endTime = performance.now();
                    
                    // Store results
                    this.comparisonPaths.set(algo, [...this.path]);
                    this.comparisonStats.set(algo, {
                        pathLength: this.calculatePathLength(),
                        time: endTime - startTime,
                        iterations: iterations
                    });
                    
                    break;
                }
                iterations++;
            }

            // Add results to table
            const stats = this.comparisonStats.get(algo);
            const row = table.insertRow();
            row.innerHTML = `
                <td><span class="legend-color" style="background: ${this.algorithmColors[algo]}"></span>${algo.toUpperCase()}</td>
                <td>${stats ? stats.pathLength.toFixed(2) : 'Failed'}</td>
                <td>${stats ? stats.time.toFixed(2) : '-'}</td>
                <td>${stats ? stats.iterations : '-'}</td>
            `;

            // Update visualization
            this.draw();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const planner = new PathPlanningBuddy();
    
    // Add event listeners for controls
    document.getElementById('start-planning').addEventListener('click', () => planner.startPlanning());
    document.getElementById('reset-planning').addEventListener('click', () => {
        planner.reset();
        planner.comparisonPaths.clear();
        planner.comparisonStats.clear();
    });
    document.getElementById('clear-obstacles').addEventListener('click', () => {
        planner.obstacles = [];
        planner.draw();
    });
    document.getElementById('generate-random').addEventListener('click', () => planner.generateRandomScene());
    document.getElementById('compare-all').addEventListener('click', () => planner.compareAllAlgorithms());
    
    // Add listeners for visualization toggles
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', () => planner.draw());
    });
}); 