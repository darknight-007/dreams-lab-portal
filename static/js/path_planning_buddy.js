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
        
        // Initialize parameters
        this.stepSize = 20;
        this.goalBias = 0.2;
        this.maxIterations = 5000;
        
        // Then setup canvas
        this.canvas = document.getElementById('path-planning-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();
        
        // Bind event listeners
        window.addEventListener('resize', () => this.resizeCanvas());
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
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
        this.isRunning = false;
        this.draw();
    }

    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Left click: Add obstacle
        if (e.button === 0) {
            this.obstacles.push({ x, y, radius: 20 });
        }
        // Right click: Set goal
        else if (e.button === 2) {
            this.goal = { x, y };
        }
        // Middle click: Set start
        else if (e.button === 1) {
            this.start = { x, y };
        }
        
        this.draw();
    }

    // RRT Implementation
    rrtStep() {
        if (this.tree.length === 0 && this.start) {
            this.tree.push([null, this.start]);
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
        if (!newNode) return false;

        // Check for collisions
        if (this.isValidPath(nearest, newNode)) {
            this.tree.push([nearest, newNode]);

            // Check if we reached the goal
            if (this.goal && this.distanceBetween(newNode, this.goal) < this.stepSize) {
                this.tree.push([newNode, this.goal]);
                return true;
            }
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
        
        // Draw path
        if (this.path.length > 0) {
            this.ctx.strokeStyle = '#4CAF50';
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

    // Add random obstacles
    generateRandomObstacles() {
        const numObstacles = 10;
        this.obstacles = [];
        for (let i = 0; i < numObstacles; i++) {
            this.obstacles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: 20 + Math.random() * 30
            });
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

        // Get parameters from UI
        this.stepSize = parseInt(document.getElementById('step-size').value);
        this.goalBias = parseInt(document.getElementById('goal-bias').value) / 100;
        this.maxIterations = parseInt(document.getElementById('max-iterations').value);

        // Start RRT
        let iterations = 0;
        const startTime = performance.now();

        while (this.isRunning && iterations < this.maxIterations) {
            if (this.rrtStep()) {
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
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const planner = new PathPlanningBuddy();
    
    // Add event listeners for controls
    document.getElementById('start-planning').addEventListener('click', () => planner.startPlanning());
    document.getElementById('reset-planning').addEventListener('click', () => planner.reset());
    document.getElementById('clear-obstacles').addEventListener('click', () => {
        planner.obstacles = [];
        planner.draw();
    });
    document.getElementById('generate-random').addEventListener('click', () => planner.generateRandomObstacles());
    
    // Add listeners for visualization toggles
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', () => planner.draw());
    });
}); 