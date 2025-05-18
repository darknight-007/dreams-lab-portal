class BundleAdjustmentDemo {
    constructor() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        try {
            // Initialize Three.js scene
            this.scene = new THREE.Scene();
            
            // Get container and set up renderer
            const container = document.getElementById('scene-canvas');
            if (!container) {
                console.error('Scene canvas element not found');
                return;
            }
            
            this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(this.renderer.domElement);
            
            // Camera controls
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.camera.position.set(5, 5, 5);
            this.controls.update();
            
            // Scene parameters
            this.numPoints = 30;
            this.numCameras = 4;
            this.learningRate = 0.1;
            this.maxIterations = 50;
            this.noiseLevel = 0.1;
            
            // Optimization state
            this.iteration = 0;
            this.isRunning = false;
            this.cameras = [];
            this.points = [];
            this.observations = [];
            this.lastError = Infinity;
            this.convergenceThreshold = 1e-6;
            this.convergenceCount = 0;
            
            // Setup
            this.setupUI();
            this.setupScene();
            this.animate();
        } catch (error) {
            console.error('Error initializing bundle adjustment demo:', error);
        }
    }

    setupUI() {
        try {
            // Button event listeners
            const resetBtn = document.getElementById('reset-btn');
            const stepBtn = document.getElementById('step-btn');
            const runBtn = document.getElementById('run-btn');
            const pauseBtn = document.getElementById('pause-btn');
            
            if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
            if (stepBtn) stepBtn.addEventListener('click', () => this.step());
            if (runBtn) runBtn.addEventListener('click', () => this.run());
            if (pauseBtn) pauseBtn.addEventListener('click', () => this.pause());
            
            // Slider event listeners
            const learningRateInput = document.getElementById('learning-rate');
            const learningRateValue = document.getElementById('learning-rate-value');
            if (learningRateInput && learningRateValue) {
                learningRateInput.addEventListener('input', (e) => {
                    this.learningRate = parseInt(e.target.value) / 100;
                    learningRateValue.textContent = this.learningRate.toFixed(2);
                });
            }
            
            const maxIterationsInput = document.getElementById('max-iterations');
            const maxIterationsValue = document.getElementById('max-iterations-value');
            if (maxIterationsInput && maxIterationsValue) {
                maxIterationsInput.addEventListener('input', (e) => {
                    this.maxIterations = parseInt(e.target.value);
                    maxIterationsValue.textContent = this.maxIterations;
                });
            }
            
            const noiseLevelInput = document.getElementById('noise-level');
            const noiseLevelValue = document.getElementById('noise-level-value');
            if (noiseLevelInput && noiseLevelValue) {
                noiseLevelInput.addEventListener('input', (e) => {
                    this.noiseLevel = parseInt(e.target.value) / 100;
                    noiseLevelValue.textContent = this.noiseLevel.toFixed(2);
                });
            }
            
            const numPointsInput = document.getElementById('num-points');
            const numPointsValue = document.getElementById('num-points-value');
            if (numPointsInput && numPointsValue) {
                numPointsInput.addEventListener('input', (e) => {
                    this.numPoints = parseInt(e.target.value);
                    numPointsValue.textContent = this.numPoints;
                    this.reset();
                });
            }
            
            // Handle window resize
            window.addEventListener('resize', () => {
                const container = document.getElementById('scene-canvas');
                if (container) {
                    this.camera.aspect = container.clientWidth / container.clientHeight;
                    this.camera.updateProjectionMatrix();
                    this.renderer.setSize(container.clientWidth, container.clientHeight);
                }
            });
        } catch (error) {
            console.error('Error setting up UI:', error);
        }
    }
    
    setupScene() {
        // Clear existing scene
        while(this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);  // Increased intensity
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);  // Increased intensity
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);

        // Add fill light from opposite direction
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-10, 5, -10);
        this.scene.add(fillLight);
        
        // Add enhanced ground grid
        const gridHelper = new THREE.GridHelper(10, 20, 0x888888, 0x444444);  // More divisions, better colors
        gridHelper.material.opacity = 0.5;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);
        
        // Generate random 3D points with enhanced visualization
        this.points = [];
        for(let i = 0; i < this.numPoints; i++) {
            const point = {
                x: (Math.random() - 0.5) * 6,
                y: Math.random() * 3,
                z: (Math.random() - 0.5) * 6
            };
            this.points.push(point);
            
            // Add point visualization with enhanced material
            const geometry = new THREE.SphereGeometry(0.12);  // Slightly larger
            const material = new THREE.MeshPhongMaterial({ 
                color: 0x00ff00,
                emissive: 0x004400,  // Slight glow
                shininess: 60,
                specular: 0x004400
            });
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(point.x, point.y, point.z);
            this.scene.add(sphere);
        }
        
        // Set up cameras with pyramidal visualization
        this.cameras = [];
        for(let i = 0; i < this.numCameras; i++) {
            const angle = (i / this.numCameras) * Math.PI * 2;
            const radius = 4;
            
            // Camera position
            const position = {
                x: Math.cos(angle) * radius,
                y: 2,
                z: Math.sin(angle) * radius
            };
            
            // Calculate rotation to look at center
            const center = new THREE.Vector3(0, 0, 0);
            const up = new THREE.Vector3(0, 1, 0);
            const cameraPos = new THREE.Vector3(position.x, position.y, position.z);
            const lookAtMatrix = new THREE.Matrix4();
            lookAtMatrix.lookAt(cameraPos, center, up);
            const quaternion = new THREE.Quaternion();
            quaternion.setFromRotationMatrix(lookAtMatrix);
            const euler = new THREE.Euler();
            euler.setFromQuaternion(quaternion);
            
            const camera = {
                position: position,
                rotation: {
                    x: euler.x,
                    y: euler.y,
                    z: euler.z
                },
                focal: 1.0
            };
            this.cameras.push(camera);
            
            // Create camera pyramid geometry
            const pyramidGeometry = new THREE.BufferGeometry();
            const focalPlaneWidth = 1.0;
            const focalPlaneHeight = 0.75;
            const pyramidLength = camera.focal;
            
            // Pyramid vertices
            const vertices = new Float32Array([
                // Pyramid faces
                0, 0, 0,  // apex
                -focalPlaneWidth/2, -focalPlaneHeight/2, -pyramidLength,  // bottom left
                focalPlaneWidth/2, -focalPlaneHeight/2, -pyramidLength,   // bottom right
                
                0, 0, 0,  // apex
                focalPlaneWidth/2, -focalPlaneHeight/2, -pyramidLength,   // bottom right
                focalPlaneWidth/2, focalPlaneHeight/2, -pyramidLength,    // top right
                
                0, 0, 0,  // apex
                focalPlaneWidth/2, focalPlaneHeight/2, -pyramidLength,    // top right
                -focalPlaneWidth/2, focalPlaneHeight/2, -pyramidLength,   // top left
                
                0, 0, 0,  // apex
                -focalPlaneWidth/2, focalPlaneHeight/2, -pyramidLength,   // top left
                -focalPlaneWidth/2, -focalPlaneHeight/2, -pyramidLength,  // bottom left
                
                // Focal plane
                -focalPlaneWidth/2, -focalPlaneHeight/2, -pyramidLength,  // bottom left
                focalPlaneWidth/2, -focalPlaneHeight/2, -pyramidLength,   // bottom right
                focalPlaneWidth/2, focalPlaneHeight/2, -pyramidLength,    // top right
                
                -focalPlaneWidth/2, -focalPlaneHeight/2, -pyramidLength,  // bottom left
                focalPlaneWidth/2, focalPlaneHeight/2, -pyramidLength,    // top right
                -focalPlaneWidth/2, focalPlaneHeight/2, -pyramidLength    // top left
            ]);
            
            pyramidGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            pyramidGeometry.computeVertexNormals();
            
            // Create materials
            const pyramidMaterial = new THREE.MeshPhongMaterial({
                color: 0xff4444,
                emissive: 0x441111,
                shininess: 70,
                specular: 0x441111,
                transparent: true,
                opacity: 0.8,
                side: THREE.DoubleSide
            });
            
            const focalPlaneMaterial = new THREE.MeshPhongMaterial({
                color: 0x4444ff,
                emissive: 0x111144,
                shininess: 70,
                specular: 0x111144,
                transparent: true,
                opacity: 0.3,
                side: THREE.DoubleSide
            });
            
            // Create meshes
            const pyramidMesh = new THREE.Mesh(pyramidGeometry, pyramidMaterial);
            pyramidMesh.position.copy(cameraPos);
            pyramidMesh.setRotationFromQuaternion(quaternion);
            this.scene.add(pyramidMesh);
            
            // Add coordinate axes for each camera
            const axesHelper = new THREE.AxesHelper(0.5);
            axesHelper.position.copy(cameraPos);
            axesHelper.setRotationFromQuaternion(quaternion);
            this.scene.add(axesHelper);
            
            // Add focal plane
            const staticFocalPlaneGeometry = new THREE.PlaneGeometry(focalPlaneWidth, focalPlaneHeight);
            const staticFocalPlaneMaterial = new THREE.MeshPhongMaterial({
                color: 0x4444ff,
                transparent: true,
                opacity: 0.2,
                side: THREE.DoubleSide
            });
            const staticFocalPlane = new THREE.Mesh(staticFocalPlaneGeometry, staticFocalPlaneMaterial);
            staticFocalPlane.position.copy(cameraPos);
            staticFocalPlane.setRotationFromQuaternion(quaternion);
            staticFocalPlane.translateZ(-pyramidLength);  // Move to focal distance
            this.scene.add(staticFocalPlane);
        }
        
        // Add projection visualization group
        this.projectionGroup = new THREE.Group();
        this.scene.add(this.projectionGroup);
        
        // Generate observations with noise
        this.generateObservations();
        this.updateProjectionVisualization();

        // Set scene background color
        this.scene.background = new THREE.Color(0x1a1a1a);  // Dark gray background
    }
    
    generateObservations() {
        this.observations = [];
        
        for(let i = 0; i < this.cameras.length; i++) {
            const camera = this.cameras[i];
            const observations = [];
            
            for(let j = 0; j < this.points.length; j++) {
                const point = this.points[j];
                
                // Project point onto camera image plane
                const projection = this.projectPoint(point, camera);
                
                // Add noise to projection
                projection.x += (Math.random() - 0.5) * this.noiseLevel;
                projection.y += (Math.random() - 0.5) * this.noiseLevel;
                
                observations.push(projection);
            }
            
            this.observations.push(observations);
        }
    }
    
    projectPoint(point, camera) {
        // Convert point to camera coordinates using lookAt matrix
        const cameraPos = new THREE.Vector3(camera.position.x, camera.position.y, camera.position.z);
        const center = new THREE.Vector3(0, 0, 0);
        const up = new THREE.Vector3(0, 1, 0);
        const lookAtMatrix = new THREE.Matrix4();
        lookAtMatrix.lookAt(cameraPos, center, up);
        lookAtMatrix.invert();  // Get camera-to-world transform
        
        // Transform point to camera space
        const pointVec = new THREE.Vector3(point.x, point.y, point.z);
        pointVec.applyMatrix4(lookAtMatrix);
        
        // Project to image plane
        return {
            x: camera.focal * pointVec.x / -pointVec.z,  // Note: negative z because camera looks down negative z-axis
            y: camera.focal * pointVec.y / -pointVec.z
        };
    }
    
    computeReprojectionError() {
        let totalError = 0;
        
        for(let i = 0; i < this.cameras.length; i++) {
            const camera = this.cameras[i];
            
            for(let j = 0; j < this.points.length; j++) {
                const point = this.points[j];
                const observation = this.observations[i][j];
                
                // Project point
                const projection = this.projectPoint(point, camera);
                
                // Compute error
                const dx = projection.x - observation.x;
                const dy = projection.y - observation.y;
                totalError += dx*dx + dy*dy;
            }
        }
        
        return Math.sqrt(totalError / (this.cameras.length * this.points.length));
    }
    
    optimizationStep() {
        // Compute gradients for cameras and points
        const cameraGradients = this.cameras.map(() => ({
            position: { x: 0, y: 0, z: 0 },
            rotation: { x: 0, y: 0, z: 0 }
        }));
        
        const pointGradients = this.points.map(() => ({
            x: 0, y: 0, z: 0
        }));
        
        // Compute gradients using finite differences
        const h = 0.0001;
        
        // For each camera
        for(let i = 0; i < this.cameras.length; i++) {
            const camera = this.cameras[i];
            
            // Position gradients
            ['x', 'y', 'z'].forEach(axis => {
                camera.position[axis] += h;
                const error1 = this.computeReprojectionError();
                camera.position[axis] -= 2*h;
                const error2 = this.computeReprojectionError();
                camera.position[axis] += h;
                
                cameraGradients[i].position[axis] = (error1 - error2) / (2*h);
            });
            
            // Rotation gradients
            ['x', 'y', 'z'].forEach(axis => {
                camera.rotation[axis] += h;
                const error1 = this.computeReprojectionError();
                camera.rotation[axis] -= 2*h;
                const error2 = this.computeReprojectionError();
                camera.rotation[axis] += h;
                
                cameraGradients[i].rotation[axis] = (error1 - error2) / (2*h);
            });
        }
        
        // For each point
        for(let i = 0; i < this.points.length; i++) {
            const point = this.points[i];
            
            // Position gradients
            ['x', 'y', 'z'].forEach(axis => {
                point[axis] += h;
                const error1 = this.computeReprojectionError();
                point[axis] -= 2*h;
                const error2 = this.computeReprojectionError();
                point[axis] += h;
                
                pointGradients[i][axis] = (error1 - error2) / (2*h);
            });
        }
        
        // Update parameters
        for(let i = 0; i < this.cameras.length; i++) {
            const camera = this.cameras[i];
            const gradient = cameraGradients[i];
            
            ['x', 'y', 'z'].forEach(axis => {
                camera.position[axis] -= this.learningRate * gradient.position[axis];
                camera.rotation[axis] -= this.learningRate * gradient.rotation[axis];
            });
        }
        
        for(let i = 0; i < this.points.length; i++) {
            const point = this.points[i];
            const gradient = pointGradients[i];
            
            ['x', 'y', 'z'].forEach(axis => {
                point[axis] -= this.learningRate * gradient[axis];
            });
        }
        
        // Update visualizations
        this.updateVisualization();
        this.updateProjectionVisualization();
        
        // Update metrics
        const error = this.computeReprojectionError();
        document.getElementById('reprojection-error').textContent = error.toFixed(4);
        document.getElementById('iteration-count').textContent = this.iteration;
        
        // Check convergence
        const errorImprovement = Math.abs(this.lastError - error);
        if(errorImprovement < this.convergenceThreshold) {
            this.convergenceCount++;
            if(this.convergenceCount > 5) {  // Require 5 consecutive small improvements
                this.pause();
                document.getElementById('status').textContent = 'Converged';
                return true;
            }
        } else {
            this.convergenceCount = 0;
        }
        this.lastError = error;
        
        // Compute camera and point errors
        const cameraError = cameraGradients.reduce((sum, g) => 
            sum + Object.values(g.position).reduce((s, v) => s + v*v, 0) +
                  Object.values(g.rotation).reduce((s, v) => s + v*v, 0)
        , 0);
        
        const pointError = pointGradients.reduce((sum, g) => 
            sum + Object.values(g).reduce((s, v) => s + v*v, 0)
        , 0);
        
        document.getElementById('camera-error').textContent = Math.sqrt(cameraError).toFixed(4);
        document.getElementById('point-error').textContent = Math.sqrt(pointError).toFixed(4);
        return false;
    }
    
    updateVisualization() {
        // Update camera visualizations
        let cameraIndex = 0;
        this.scene.children.forEach(child => {
            if(child instanceof THREE.Mesh && child.geometry instanceof THREE.ConeGeometry ||
               child instanceof THREE.LineSegments) {
                const camera = this.cameras[Math.floor(cameraIndex / 2)];
                child.position.set(camera.position.x, camera.position.y, camera.position.z);
                child.rotation.set(camera.rotation.x, camera.rotation.y, camera.rotation.z);
                cameraIndex++;
            }
        });
        
        // Update point visualizations
        let sphereIndex = 0;
        this.scene.children.forEach(child => {
            if(child instanceof THREE.Mesh && child.geometry instanceof THREE.SphereGeometry) {
                const point = this.points[sphereIndex++];
                child.position.set(point.x, point.y, point.z);
            }
        });
    }
    
    updateProjectionVisualization() {
        // Clear existing projection lines
        while(this.projectionGroup.children.length > 0) {
            this.projectionGroup.remove(this.projectionGroup.children[0]);
        }
        
        // Draw projection lines and points for each camera-point pair
        for(let i = 0; i < this.cameras.length; i++) {
            const camera = this.cameras[i];
            const cameraPos = new THREE.Vector3(camera.position.x, camera.position.y, camera.position.z);
            const center = new THREE.Vector3(0, 0, 0);
            const up = new THREE.Vector3(0, 1, 0);
            const lookAtMatrix = new THREE.Matrix4();
            lookAtMatrix.lookAt(cameraPos, center, up);
            const quaternion = new THREE.Quaternion();
            quaternion.setFromRotationMatrix(lookAtMatrix);
            
            // Create focal plane for projected points
            const dynamicFocalPlaneGeometry = new THREE.PlaneGeometry(1.0, 0.75);
            const dynamicFocalPlaneMaterial = new THREE.MeshPhongMaterial({
                color: 0x4444ff,
                transparent: true,
                opacity: 0.2,
                side: THREE.DoubleSide
            });
            const dynamicFocalPlane = new THREE.Mesh(dynamicFocalPlaneGeometry, dynamicFocalPlaneMaterial);
            dynamicFocalPlane.position.copy(cameraPos);
            dynamicFocalPlane.setRotationFromQuaternion(quaternion);
            dynamicFocalPlane.translateZ(-camera.focal);
            this.projectionGroup.add(dynamicFocalPlane);
            
            for(let j = 0; j < this.points.length; j++) {
                const point = this.points[j];
                const observation = this.observations[i][j];
                const projection = this.projectPoint(point, camera);
                
                // Draw line from camera to point
                const lineGeometry = new THREE.BufferGeometry();
                const vertices = new Float32Array([
                    camera.position.x, camera.position.y, camera.position.z,
                    point.x, point.y, point.z
                ]);
                lineGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                
                const lineMaterial = new THREE.LineBasicMaterial({
                    color: 0x4444ff,
                    opacity: 0.4,
                    transparent: true,
                    linewidth: 1
                });
                
                const line = new THREE.Line(lineGeometry, lineMaterial);
                this.projectionGroup.add(line);
                
                // Add projected point on focal plane
                const projectedPointGeometry = new THREE.SphereGeometry(0.02);
                const projectedPointMaterial = new THREE.MeshPhongMaterial({
                    color: 0x44ff44,
                    emissive: 0x114411
                });
                const projectedPoint = new THREE.Mesh(projectedPointGeometry, projectedPointMaterial);
                
                // Convert projection to 3D coordinates on focal plane
                const worldProjection = new THREE.Vector3(
                    projection.x * 0.1,  // Scale down the projection coordinates
                    projection.y * 0.1,
                    -camera.focal
                );
                worldProjection.applyEuler(new THREE.Euler(
                    camera.rotation.x,
                    camera.rotation.y,
                    camera.rotation.z
                ));
                worldProjection.add(new THREE.Vector3(
                    camera.position.x,
                    camera.position.y,
                    camera.position.z
                ));
                
                projectedPoint.position.copy(worldProjection);
                this.projectionGroup.add(projectedPoint);
                
                // Draw reprojection error if significant
                const error = Math.sqrt(
                    Math.pow(projection.x - observation.x, 2) +
                    Math.pow(projection.y - observation.y, 2)
                );
                
                if(error > 0.1) {
                    const errorMaterial = new THREE.LineBasicMaterial({
                        color: 0xff4444,
                        opacity: Math.min(error / 1.5, 1),
                        transparent: true,
                        linewidth: 2
                    });
                    
                    // Convert observation to 3D coordinates
                    const worldObservation = new THREE.Vector3(
                        observation.x * 0.1,
                        observation.y * 0.1,
                        -camera.focal
                    );
                    worldObservation.applyEuler(new THREE.Euler(
                        camera.rotation.x,
                        camera.rotation.y,
                        camera.rotation.z
                    ));
                    worldObservation.add(new THREE.Vector3(
                        camera.position.x,
                        camera.position.y,
                        camera.position.z
                    ));
                    
                    const errorGeometry = new THREE.BufferGeometry();
                    const errorVertices = new Float32Array([
                        worldProjection.x, worldProjection.y, worldProjection.z,
                        worldObservation.x, worldObservation.y, worldObservation.z
                    ]);
                    errorGeometry.setAttribute('position', new THREE.BufferAttribute(errorVertices, 3));
                    const errorLine = new THREE.Line(errorGeometry, errorMaterial);
                    this.projectionGroup.add(errorLine);
                }
            }
        }
    }
    
    reset() {
        this.iteration = 0;
        this.isRunning = false;
        this.lastError = Infinity;
        this.convergenceCount = 0;
        document.getElementById('run-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;
        document.getElementById('status').textContent = 'Ready';
        
        this.setupScene();
        
        // Reset metrics display
        document.getElementById('reprojection-error').textContent = '-';
        document.getElementById('iteration-count').textContent = '0';
        document.getElementById('camera-error').textContent = '-';
        document.getElementById('point-error').textContent = '-';
    }
    
    step() {
        if(this.iteration < this.maxIterations) {
            this.iteration++;
            this.optimizationStep();
        }
    }
    
    async run() {
        this.isRunning = true;
        document.getElementById('run-btn').disabled = true;
        document.getElementById('pause-btn').disabled = false;
        document.getElementById('status').textContent = 'Running';
        
        while(this.isRunning && this.iteration < this.maxIterations) {
            const converged = this.step();
            if(converged) break;
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        if(this.iteration >= this.maxIterations) {
            this.pause();
            document.getElementById('status').textContent = 'Max iterations reached';
        }
    }
    
    pause() {
        this.isRunning = false;
        document.getElementById('run-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    updateMetrics(error) {
        try {
            const elements = {
                'reprojection-error': error.toFixed(4),
                'iteration-count': this.iteration,
                'camera-error': Math.sqrt(this.cameraError).toFixed(4),
                'point-error': Math.sqrt(this.pointError).toFixed(4),
                'status': this.status
            };

            Object.entries(elements).forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element) element.textContent = value;
            });
        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    }
}

// Initialize demo when the page loads
window.addEventListener('load', () => {
    try {
        const demo = new BundleAdjustmentDemo();
    } catch (error) {
        console.error('Error creating bundle adjustment demo:', error);
    }
});
