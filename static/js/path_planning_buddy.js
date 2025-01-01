class PathPlanningBuddy {
    constructor() {
        this.canvas = document.getElementById('planning-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();

        // State
        this.obstacles = [];
        this.startPoint = null;
        this.goalPoint = null;
        this.rrtTree = [];
        this.path = [];
        this.samples = [];
        this.isPlanning = false;
        this.isPaused = false;
        this.currentAlgorithm = 'rrt';

        // RRT Parameters
        this.stepSize = 20;
        this.goalBias = 0.2;
        this.maxIterations = 5000;

        // Visualization flags
        this.showTree = true;
        this.showSamples = true;
        this.showPath = true;
        this.animate = true;

        // Bind event listeners
        this.bindEventListeners();
        
        // Initialize
        this.setupCanvas();
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    bindEventListeners() {
        // Algorithm selection
        document.getElementById('algorithm-type').addEventListener('change', (e) => {
            this.currentAlgorithm = e.target.value;
            this.updateUIForAlgorithm();
        });

        // Canvas interactions
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));

        // Control buttons
        document.getElementById('start-planning').addEventListener('click', () => this.startPlanning());
        document.getElementById('pause-planning').addEventListener('click', () => this.togglePause());
        document.getElementById('reset-planning').addEventListener('click', () => this.reset());
        document.getElementById('step-planning').addEventListener('click', () => this.step());
        document.getElementById('clear-obstacles').addEventListener('click', () => this.clearObstacles());
        document.getElementById('add-random-obstacles').addEventListener('click', () => this.addRandomObstacles());
        document.getElementById('add-maze').addEventListener('click', () => this.generateMaze());

        // Parameter controls
        document.getElementById('step-size').addEventListener('input', (e) => {
            this.stepSize = parseInt(e.target.value);
        });
        document.getElementById('goal-bias').addEventListener('input', (e) => {
            this.goalBias = parseInt(e.target.value) / 100;
        });
        document.getElementById('max-iterations').addEventListener('input', (e) => {
            this.maxIterations = parseInt(e.target.value);
        });

        // Visualization controls
        document.getElementById('show-tree').addEventListener('change', (e) => {
            this.showTree = e.target.checked;
            this.draw();
        });
        document.getElementById('show-samples').addEventListener('change', (e) => {
            this.showSamples = e.target.checked;
            this.draw();
        });
        document.getElementById('show-path').addEventListener('change', (e) => {
            this.showPath = e.target.checked;
            this.draw();
        });
        document.getElementById('animate').addEventListener('change', (e) => {
            this.animate = e.target.checked;
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.draw();
        });
    }

    setupCanvas() {
        this.resizeCanvas();
        this.ctx.lineWidth = 2;
        this.draw();
    }

    // RRT Implementation
    rrtStep() {
        if (this.rrtTree.length === 0) {
            this.rrtTree.push({
                x: this.startPoint.x,
                y: this.startPoint.y,
                parent: null
            });
        }

        // Sample point
        let sample;
        if (Math.random() < this.goalBias) {
            sample = { x: this.goalPoint.x, y: this.goalPoint.y };
        } else {
            sample = {
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height
            };
        }
        this.samples.push(sample);

        // Find nearest node
        const nearest = this.findNearestNode(sample);

        // Extend tree
        const newNode = this.extend(nearest, sample);
        if (newNode && !this.checkCollision(nearest, newNode)) {
            this.rrtTree.push({
                x: newNode.x,
                y: newNode.y,
                parent: nearest
            });

            // Check if we reached the goal
            if (this.distanceBetween(newNode, this.goalPoint) < this.stepSize) {
                this.rrtTree.push({
                    x: this.goalPoint.x,
                    y: this.goalPoint.y,
                    parent: newNode
                });
                return true;
            }
        }

        return false;
    }

    findNearestNode(point) {
        let nearest = this.rrtTree[0];
        let minDist = this.distanceBetween(point, nearest);

        for (const node of this.rrtTree) {
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

    distanceBetween(p1, p2) {
        return Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
    }

    checkCollision(p1, p2) {
        for (const obstacle of this.obstacles) {
            if (this.lineIntersectsRectangle(p1, p2, obstacle)) {
                return true;
            }
        }
        return false;
    }

    lineIntersectsRectangle(p1, p2, rect) {
        const lines = [
            [{x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y}],
            [{x: rect.x + rect.width, y: rect.y}, {x: rect.x + rect.width, y: rect.y + rect.height}],
            [{x: rect.x + rect.width, y: rect.y + rect.height}, {x: rect.x, y: rect.y + rect.height}],
            [{x: rect.x, y: rect.y + rect.height}, {x: rect.x, y: rect.y}]
        ];

        for (const line of lines) {
            if (this.lineIntersectsLine(p1, p2, line[0], line[1])) {
                return true;
            }
        }

        return false;
    }

    lineIntersectsLine(p1, p2, p3, p4) {
        const denominator = ((p4.y - p3.y) * (p2.x - p1.x)) - ((p4.x - p3.x) * (p2.y - p1.y));
        if (denominator === 0) return false;

        const ua = (((p4.x - p3.x) * (p1.y - p3.y)) - ((p4.y - p3.y) * (p1.x - p3.x))) / denominator;
        const ub = (((p2.x - p1.x) * (p1.y - p3.y)) - ((p2.y - p1.y) * (p1.x - p3.x))) / denominator;

        return (ua >= 0 && ua <= 1) && (ub >= 0 && ub <= 1);
    }

    reconstructPath() {
        this.path = [];
        let current = this.rrtTree[this.rrtTree.length - 1];
        while (current) {
            this.path.unshift(current);
            current = current.parent;
        }
    }

    // RRT* Implementation
    rrtStarStep() {
        if (this.rrtTree.length === 0) {
            this.rrtTree.push({
                x: this.startPoint.x,
                y: this.startPoint.y,
                parent: null,
                cost: 0
            });
        }

        // Sample point
        let sample;
        if (Math.random() < this.goalBias) {
            sample = { x: this.goalPoint.x, y: this.goalPoint.y };
        } else {
            sample = {
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height
            };
        }
        this.samples.push(sample);

        // Find nearest node
        const nearest = this.findNearestNode(sample);

        // Extend tree
        const newNode = this.extend(nearest, sample);
        if (newNode && !this.checkCollision(nearest, newNode)) {
            // Find nearby nodes for rewiring
            const radius = Math.min(this.stepSize * 5, 
                Math.max(20, 50 * Math.log(this.rrtTree.length) / this.rrtTree.length));
            const nearbyNodes = this.findNodesInRadius(newNode, radius);

            // Connect to best parent
            let bestParent = nearest;
            let bestCost = nearest.cost + this.distanceBetween(nearest, newNode);

            for (const node of nearbyNodes) {
                const potentialCost = node.cost + this.distanceBetween(node, newNode);
                if (potentialCost < bestCost && !this.checkCollision(node, newNode)) {
                    bestParent = node;
                    bestCost = potentialCost;
                }
            }

            // Add new node
            const addedNode = {
                x: newNode.x,
                y: newNode.y,
                parent: bestParent,
                cost: bestCost
            };
            this.rrtTree.push(addedNode);

            // Rewire nearby nodes
            for (const node of nearbyNodes) {
                const potentialCost = bestCost + this.distanceBetween(addedNode, node);
                if (potentialCost < node.cost && !this.checkCollision(addedNode, node)) {
                    node.parent = addedNode;
                    node.cost = potentialCost;
                    this.updateDescendantsCosts(node);
                }
            }

            // Check if we reached the goal
            if (this.distanceBetween(newNode, this.goalPoint) < this.stepSize) {
                const finalCost = bestCost + this.distanceBetween(newNode, this.goalPoint);
                this.rrtTree.push({
                    x: this.goalPoint.x,
                    y: this.goalPoint.y,
                    parent: addedNode,
                    cost: finalCost
                });
                return true;
            }
        }

        return false;
    }

    findNodesInRadius(point, radius) {
        return this.rrtTree.filter(node => 
            this.distanceBetween(node, point) <= radius);
    }

    updateDescendantsCosts(node) {
        const descendants = this.rrtTree.filter(n => n.parent === node);
        for (const descendant of descendants) {
            descendant.cost = node.cost + this.distanceBetween(node, descendant);
            this.updateDescendantsCosts(descendant);
        }
    }

    // Planning Control
    async startPlanning() {
        if (!this.startPoint || !this.goalPoint) {
            alert('Please set start and goal points first');
            return;
        }

        this.isPlanning = true;
        this.isPaused = false;
        
        let iterations = 0;
        const startTime = performance.now();

        while (this.isPlanning && !this.isPaused && iterations < this.maxIterations) {
            let success = false;
            
            switch (this.currentAlgorithm) {
                case 'rrt':
                    success = this.rrtStep();
                    break;
                case 'rrt-star':
                    success = this.rrtStarStep();
                    break;
                // Add other algorithms here
            }

            if (success) {
                this.reconstructPath();
                this.updateStatistics(iterations, startTime);
                break;
            }
            iterations++;

            if (this.animate) {
                this.draw();
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        if (!this.path.length && iterations >= this.maxIterations) {
            alert('Maximum iterations reached without finding a path');
        }

        this.draw();
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        if (!this.isPaused) this.startPlanning();
    }

    reset() {
        this.rrtTree = [];
        this.path = [];
        this.samples = [];
        this.isPlanning = false;
        this.isPaused = false;
        this.draw();
    }

    step() {
        if (this.rrtStep()) {
            this.reconstructPath();
        }
        this.draw();
    }

    // Environment Modification
    clearObstacles() {
        this.obstacles = [];
        this.draw();
    }

    addRandomObstacles() {
        for (let i = 0; i < 10; i++) {
            this.obstacles.push({
                x: Math.random() * (this.canvas.width - 50),
                y: Math.random() * (this.canvas.height - 50),
                width: 30 + Math.random() * 40,
                height: 30 + Math.random() * 40
            });
        }
        this.draw();
    }

    generateMaze() {
        // Simple maze generation
        const cellSize = 40;
        const cols = Math.floor(this.canvas.width / cellSize);
        const rows = Math.floor(this.canvas.height / cellSize);

        this.obstacles = [];
        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < rows; j++) {
                if (Math.random() < 0.3) {
                    this.obstacles.push({
                        x: i * cellSize,
                        y: j * cellSize,
                        width: cellSize - 2,
                        height: cellSize - 2
                    });
                }
            }
        }
        this.draw();
    }

    // Drawing
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw obstacles
        this.ctx.fillStyle = '#666';
        for (const obstacle of this.obstacles) {
            this.ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
        }

        // Draw RRT tree with cost coloring for RRT*
        if (this.showTree) {
            for (const node of this.rrtTree) {
                if (node.parent) {
                    // Color based on cost for RRT*
                    if (this.currentAlgorithm === 'rrt-star' && node.cost !== undefined) {
                        const maxCost = Math.max(...this.rrtTree.map(n => n.cost || 0));
                        const normalizedCost = node.cost / maxCost;
                        const hue = (1 - normalizedCost) * 240; // Blue (240) to Red (0)
                        this.ctx.strokeStyle = `hsl(${hue}, 70%, 50%)`;
                    } else {
                        this.ctx.strokeStyle = '#aaa';
                    }
                    
                    this.ctx.beginPath();
                    this.ctx.moveTo(node.parent.x, node.parent.y);
                    this.ctx.lineTo(node.x, node.y);
                    this.ctx.stroke();
                }
            }
        }

        // Draw samples
        if (this.showSamples) {
            this.ctx.fillStyle = '#ccc';
            for (const sample of this.samples) {
                this.ctx.beginPath();
                this.ctx.arc(sample.x, sample.y, 2, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }

        // Draw path with gradient
        if (this.showPath && this.path.length > 0) {
            const gradient = this.ctx.createLinearGradient(
                this.path[0].x, this.path[0].y,
                this.path[this.path.length - 1].x, this.path[this.path.length - 1].y
            );
            gradient.addColorStop(0, '#0f0');   // Start: Green
            gradient.addColorStop(0.5, '#ff0');  // Middle: Yellow
            gradient.addColorStop(1, '#f00');    // End: Red
            
            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(this.path[0].x, this.path[0].y);
            for (let i = 1; i < this.path.length; i++) {
                this.ctx.lineTo(this.path[i].x, this.path[i].y);
            }
            this.ctx.stroke();
            this.ctx.lineWidth = 2;
        }

        // Draw start and goal with glow effect
        if (this.startPoint) {
            this.ctx.shadowColor = '#0f0';
            this.ctx.shadowBlur = 15;
            this.ctx.fillStyle = '#0f0';
            this.ctx.beginPath();
            this.ctx.arc(this.startPoint.x, this.startPoint.y, 8, 0, Math.PI * 2);
            this.ctx.fill();
        }
        if (this.goalPoint) {
            this.ctx.shadowColor = '#f00';
            this.ctx.shadowBlur = 15;
            this.ctx.fillStyle = '#f00';
            this.ctx.beginPath();
            this.ctx.arc(this.goalPoint.x, this.goalPoint.y, 8, 0, Math.PI * 2);
            this.ctx.fill();
        }
        this.ctx.shadowBlur = 0;
    }

    // Mouse Interaction
    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (e.shiftKey) {
            // Start drawing obstacle
            this.isDrawingObstacle = true;
            this.currentObstacle = { x, y, width: 0, height: 0 };
        } else if (e.button === 0) {
            // Left click - set start point
            this.startPoint = { x, y };
        } else if (e.button === 2) {
            // Right click - set goal point
            this.goalPoint = { x, y };
        }
        this.draw();
    }

    handleMouseMove(e) {
        if (!this.isDrawingObstacle) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        this.currentObstacle.width = x - this.currentObstacle.x;
        this.currentObstacle.height = y - this.currentObstacle.y;
        this.draw();

        // Draw current obstacle
        this.ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
        this.ctx.fillRect(
            this.currentObstacle.x,
            this.currentObstacle.y,
            this.currentObstacle.width,
            this.currentObstacle.height
        );
    }

    handleMouseUp(e) {
        if (this.isDrawingObstacle) {
            this.isDrawingObstacle = false;
            if (Math.abs(this.currentObstacle.width) > 5 && Math.abs(this.currentObstacle.height) > 5) {
                // Normalize rectangle coordinates
                const obstacle = {
                    x: this.currentObstacle.width > 0 ? this.currentObstacle.x : this.currentObstacle.x + this.currentObstacle.width,
                    y: this.currentObstacle.height > 0 ? this.currentObstacle.y : this.currentObstacle.y + this.currentObstacle.height,
                    width: Math.abs(this.currentObstacle.width),
                    height: Math.abs(this.currentObstacle.height)
                };
                this.obstacles.push(obstacle);
            }
            this.currentObstacle = null;
            this.draw();
        }
    }

    // UI Updates
    updateUIForAlgorithm() {
        const rrtParams = document.querySelector('.rrt-params');
        const astarParams = document.querySelector('.astar-params');
        const potentialParams = document.querySelector('.potential-params');

        rrtParams.style.display = this.currentAlgorithm.startsWith('rrt') ? 'block' : 'none';
        astarParams.style.display = this.currentAlgorithm === 'astar' ? 'block' : 'none';
        potentialParams.style.display = this.currentAlgorithm === 'potential' ? 'block' : 'none';
    }

    updateStatistics(iterations, startTime) {
        const time = performance.now() - startTime;
        document.getElementById('computation-time').textContent = `Time: ${time.toFixed(2)} ms`;
        document.getElementById('nodes-explored').textContent = `Nodes Explored: ${this.rrtTree.length}`;
        
        let pathLength = 0;
        for (let i = 1; i < this.path.length; i++) {
            pathLength += this.distanceBetween(this.path[i-1], this.path[i]);
        }
        document.getElementById('path-length').textContent = `Path Length: ${pathLength.toFixed(2)}`;
    }
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const planner = new PathPlanningBuddy();
}); 