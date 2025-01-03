import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class MultiviewGeometry {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.camera1 = null;
        this.camera2 = null;
        this.point3D = null;
        this.gridHelper = null;
        
        this.params = {
            focalLength: 35,
            baseline: 50,
            pointDepth: 1000,
            toeIn: 0,
            principalPoint: { x: 0, y: 0 },
            sensorSize: { width: 36, height: 24 }, // 35mm format
            minBaseline: 10, // Minimum baseline in cm
            maxBaselineRatio: 0.5 // Maximum baseline as ratio of point depth
        };

        this.views = {
            top: new THREE.Vector3(0, 10, 0),
            front: new THREE.Vector3(0, 0, 10),
            isometric: new THREE.Vector3(5, 5, 5)
        };

        this.features = [];
        this.convergencePoint = null;
        this.projectionPlanes = { left: null, right: null };
        this.featureProjections = { left: [], right: [] };
        
        this.init();
        this.setupEventListeners();
        this.animate();
    }

    init() {
        // Initialize Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        this.scene.add(directionalLight);

        // Setup camera
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.copy(this.views.isometric);
        this.camera.lookAt(0, 0, 0);

        // Setup renderer
        const container = document.getElementById('three-container');
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        // Add orbit controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Add grid helper
        this.gridHelper = new THREE.GridHelper(10, 10);
        this.scene.add(this.gridHelper);

        // Create stereo cameras
        this.camera1 = this.createCameraModel();
        this.camera2 = this.createCameraModel();
        this.scene.add(this.camera1);
        this.scene.add(this.camera2);

        // Create 3D point
        const pointGeometry = new THREE.SphereGeometry(0.1);
        const pointMaterial = new THREE.MeshPhongMaterial({ color: 0xff0000 });
        this.point3D = new THREE.Mesh(pointGeometry, pointMaterial);
        this.scene.add(this.point3D);

        // Add convergence point sphere
        const sphereGeometry = new THREE.SphereGeometry(0.15);
        const sphereMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xffff00,
            transparent: true,
            opacity: 0.7
        });
        this.convergencePoint = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.scene.add(this.convergencePoint);

        // Create random 3D features
        this.createRandomFeatures(10);

        // Create projection planes with renderers
        this.createProjectionPlanes();

        // Initial update
        this.updateScene();
    }

    createCameraModel() {
        const cameraGroup = new THREE.Group();

        // Camera body
        const bodyGeometry = new THREE.BoxGeometry(0.2, 0.2, 0.3);
        const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x808080 });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        cameraGroup.add(body);

        // Camera frustum
        const frustumGeometry = new THREE.ConeGeometry(0.3, 0.5, 4);
        const frustumMaterial = new THREE.MeshPhongMaterial({
            color: 0x404040,
            transparent: true,
            opacity: 0.5
        });
        const frustum = new THREE.Mesh(frustumGeometry, frustumMaterial);
        frustum.rotation.x = Math.PI / 2;
        frustum.position.z = 0.4;
        cameraGroup.add(frustum);

        // Camera coordinate axes
        const axesHelper = new THREE.AxesHelper(0.3);
        cameraGroup.add(axesHelper);

        // Add image plane
        const planeGeometry = new THREE.PlaneGeometry(0.4, 0.3);
        const planeMaterial = new THREE.MeshPhongMaterial({
            color: 0xcccccc,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide
        });
        const imagePlane = new THREE.Mesh(planeGeometry, planeMaterial);
        imagePlane.position.z = 0.6;
        cameraGroup.add(imagePlane);

        return cameraGroup;
    }

    updateScene() {
        const { focalLength, baseline, pointDepth, toeIn, minBaseline, maxBaselineRatio } = this.params;
        
        // Enforce minimum baseline and maximum based on point depth
        const maxBaseline = pointDepth * maxBaselineRatio;
        const safeBaseline = Math.max(minBaseline, Math.min(baseline, maxBaseline));
        
        // Convert baseline to meters for THREE.js scene
        const baselineMeters = safeBaseline / 100;
        
        // Update camera positions with safe baseline (in meters)
        this.camera1.position.set(-baselineMeters/2, 0, 0);
        this.camera2.position.set(baselineMeters/2, 0, 0);
        
        // Calculate convergence point with safety checks
        const toeInRad = (toeIn * Math.PI) / 180;
        
        // Update camera rotations using toe-in angle
        this.camera1.rotation.set(0, toeInRad, 0);
        this.camera2.rotation.set(0, -toeInRad, 0);
        
        if (Math.abs(toeInRad) > 0.001) {  // Non-zero toe-in
            // Convert baseline to meters for convergence calculation
            const convergenceDistance = (baselineMeters/2) / Math.tan(Math.abs(toeInRad));
            // Limit convergence distance and convert point depth to meters
            const safeDistance = Math.min(convergenceDistance, pointDepth/100 * 2);
            this.convergencePoint.position.set(0, 0, -safeDistance);
            this.convergencePoint.visible = true;
        } else {
            // Convert point depth to meters
            this.convergencePoint.position.set(0, 0, -pointDepth/100);
            this.convergencePoint.visible = false;
        }

        // Update feature projections
        this.updateFeatureProjections();
        this.updateMatrices();

        // Update camera views
        this.updateCameraViews();
    }

    updateFeatureProjections() {
        try {
            // Clear previous projections
            this.featureProjections.left = [];
            this.featureProjections.right = [];

            // Project features onto each camera's image plane
            this.features.forEach(feature => {
                // Get feature depth relative to camera
                const featurePos = feature.position.clone();
                const depth = -featurePos.z;
                
                // Skip features that are too close or too far
                if (depth < 0.1 || depth > this.params.pointDepth * 2) {
                    feature.visible = false;
                    return;
                }
                
                // Scale feature size based on depth
                const scale = 0.1 / Math.max(1, Math.sqrt(depth));
                feature.scale.setScalar(scale);
                feature.visible = true;

                // Project to left camera
                const leftProj = this.projectToCamera(featurePos, this.camera1, this.projectionPlanes.left);
                if (leftProj && this.isFeatureVisible(leftProj)) {
                    this.featureProjections.left.push(leftProj);
                }

                // Project to right camera
                const rightProj = this.projectToCamera(featurePos, this.camera2, this.projectionPlanes.right);
                if (rightProj && this.isFeatureVisible(rightProj)) {
                    this.featureProjections.right.push(rightProj);
                }
            });

            // Update projection plane textures
            this.updateProjectionPlaneTextures();
        } catch (error) {
            console.error('Error in updateFeatureProjections:', error);
        }
    }

    isFeatureVisible(projection) {
        const { sensorSize } = this.params;
        // Check if projection is within sensor bounds with margin
        const margin = 0.1;
        return Math.abs(projection.x) <= (sensorSize.width/2 + margin) &&
               Math.abs(projection.y) <= (sensorSize.height/2 + margin) &&
               projection.z < 0;  // Check if in front of camera
    }

    projectToCamera(point, camera, projectionPlane) {
        try {
            // Transform point to camera space
            const cameraWorldMatrix = camera.matrixWorld.clone();
            const cameraPosition = new THREE.Vector3();
            camera.getWorldPosition(cameraPosition);
            
            const pointInCameraSpace = point.clone()
                .sub(cameraPosition)
                .applyMatrix4(cameraWorldMatrix.invert());

            // Add epsilon to avoid division by zero
            const epsilon = 1e-6;
            if (Math.abs(pointInCameraSpace.z) < epsilon) {
                return null;
            }

            // Robust projection calculation
            const z = -pointInCameraSpace.z;  // Make sure z is positive for points in front
            const { focalLength, principalPoint, sensorSize } = this.params;
            const x = -(pointInCameraSpace.x * focalLength) / (z + epsilon) + principalPoint.x;
            const y = -(pointInCameraSpace.y * focalLength) / (z + epsilon) + principalPoint.y;

            // Check if projection is within sensor bounds with margin
            const margin = 0.1;
            if (Math.abs(x) > (sensorSize.width/2 + margin) || 
                Math.abs(y) > (sensorSize.height/2 + margin)) {
                return null;
            }

            return { x, y, z };
        } catch (error) {
            console.error('Error in projectToCamera:', error);
            return null;
        }
    }

    updateProjectionPlaneTextures() {
        try {
            // Clean up old correspondence lines and container
            const oldContainer = document.getElementById('correspondence-lines-container');
            if (oldContainer) {
                oldContainer.remove();
            }

            Object.entries(this.projectionPlanes).forEach(([side, plane]) => {
                if (!plane || !plane.canvas) {
                    console.warn(`Missing projection plane elements for ${side} camera`);
                    return;
                }

                const ctx = plane.canvas.getContext('2d');
                if (!ctx) {
                    console.warn(`Could not get 2D context for ${side} camera`);
                    return;
                }

                ctx.clearRect(0, 0, plane.canvas.width, plane.canvas.height);

                // Draw grid lines
                ctx.strokeStyle = '#cccccc';
                ctx.lineWidth = 1;
                const gridSize = 20;
                for (let i = 0; i <= plane.canvas.width; i += gridSize) {
                    ctx.beginPath();
                    ctx.moveTo(i, 0);
                    ctx.lineTo(i, plane.canvas.height);
                    ctx.stroke();
                }
                for (let i = 0; i <= plane.canvas.height; i += gridSize) {
                    ctx.beginPath();
                    ctx.moveTo(0, i);
                    ctx.lineTo(plane.canvas.width, i);
                    ctx.stroke();
                }

                // Draw principal point
                const { principalPoint, sensorSize } = this.params;
                const ppX = (principalPoint.x / sensorSize.width + 0.5) * plane.canvas.width;
                const ppY = (principalPoint.y / sensorSize.height + 0.5) * plane.canvas.height;
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 2;
                const crossSize = 10;
                ctx.beginPath();
                ctx.moveTo(ppX - crossSize, ppY);
                ctx.lineTo(ppX + crossSize, ppY);
                ctx.moveTo(ppX, ppY - crossSize);
                ctx.lineTo(ppX, ppY + crossSize);
                ctx.stroke();

                // Draw feature projections
                this.features.forEach((feature, index) => {
                    const featurePos = feature.position;
                    const proj = this.projectToCamera(featurePos, side === 'left' ? this.camera1 : this.camera2, plane);
                    
                    if (proj) {
                        // Convert to normalized image coordinates
                        const x = (proj.x / sensorSize.width + 0.5) * plane.canvas.width;
                        const y = (proj.y / sensorSize.height + 0.5) * plane.canvas.height;

                        // Draw feature point
                        ctx.fillStyle = `hsl(${(index * 30) % 360}, 100%, 50%)`;
                        ctx.beginPath();
                        ctx.arc(x, y, 4, 0, Math.PI * 2);
                        ctx.fill();

                        // Draw feature ID
                        ctx.fillStyle = '#000000';
                        ctx.font = '12px Arial';
                        ctx.fillText(index + 1, x + 5, y - 5);
                    }
                });

                // Update texture
                if (plane.mesh && plane.mesh.material && plane.mesh.material.map) {
                    plane.mesh.material.map.needsUpdate = true;
                }
            });

            // Draw correspondence lines between views
            const leftView = document.getElementById('leftView');
            const rightView = document.getElementById('rightView');
            if (leftView && rightView && this.featureProjections.left && this.featureProjections.right) {
                const container = document.createElement('div');
                container.id = 'correspondence-lines-container';
                document.body.appendChild(container);

                this.featureProjections.left.forEach((leftProj, index) => {
                    const rightProj = this.featureProjections.right[index];
                    if (leftProj && rightProj) {
                        const leftX = (leftProj.x / this.params.sensorSize.width + 0.5) * this.projectionPlanes.left.canvas.width;
                        const rightX = (rightProj.x / this.params.sensorSize.width + 0.5) * this.projectionPlanes.right.canvas.width;
                        const leftY = (leftProj.y / this.params.sensorSize.height + 0.5) * this.projectionPlanes.left.canvas.height;
                        const rightY = (rightProj.y / this.params.sensorSize.height + 0.5) * this.projectionPlanes.right.canvas.height;

                        const leftRect = leftView.getBoundingClientRect();
                        const rightRect = rightView.getBoundingClientRect();

                        const line = document.createElement('div');
                        line.className = 'correspondence-line';
                        line.style.position = 'absolute';
                        line.style.pointerEvents = 'none';
                        line.style.zIndex = '1000';
                        container.appendChild(line);

                        const x1 = leftRect.left + leftX;
                        const y1 = leftRect.top + leftY;
                        const x2 = rightRect.left + rightX;
                        const y2 = rightRect.top + rightY;

                        const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
                        const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

                        line.style.width = `${length}px`;
                        line.style.height = '2px';
                        line.style.backgroundColor = `hsla(${(index * 30) % 360}, 100%, 50%, 0.5)`;
                        line.style.transform = `translate(${x1}px, ${y1}px) rotate(${angle}deg)`;
                        line.style.transformOrigin = '0 0';
                    }
                });
            }
        } catch (error) {
            console.error('Error in updateProjectionPlaneTextures:', error);
        }
    }

    // Helper function to multiply matrix with vector
    multiplyMatrixVector(matrix, vector) {
        return matrix.map(row => 
            row.reduce((sum, val, i) => sum + val * vector[i], 0)
        );
    }

    // Helper function to find line intersections with image boundaries
    findLineImageIntersections(a, b, c, width, height) {
        const intersections = [];
        
        // Check intersection with left boundary (x = 0)
        const y1 = -c / b;
        if (y1 >= 0 && y1 <= height) {
            intersections.push({ x: 0, y: y1 });
        }
        
        // Check intersection with right boundary (x = width)
        const y2 = -(a * width + c) / b;
        if (y2 >= 0 && y2 <= height) {
            intersections.push({ x: width, y: y2 });
        }
        
        // Check intersection with top boundary (y = 0)
        const x1 = -c / a;
        if (x1 >= 0 && x1 <= width && intersections.length < 2) {
            intersections.push({ x: x1, y: 0 });
        }
        
        // Check intersection with bottom boundary (y = height)
        const x2 = -(b * height + c) / a;
        if (x2 >= 0 && x2 <= width && intersections.length < 2) {
            intersections.push({ x: x2, y: height });
        }
        
        return intersections;
    }

    // Helper function to get calibration matrix
    getCalibrationMatrix() {
        const { focalLength, principalPoint, sensorSize } = this.params;
        return [
            [focalLength, 0, principalPoint.x + sensorSize.width/2],
            [0, focalLength, principalPoint.y + sensorSize.height/2],
            [0, 0, 1]
        ];
    }

    updateMatrices() {
        const { focalLength, baseline, principalPoint, sensorSize } = this.params;

        // Calculate calibration matrix with principal point (using mm)
        const K = [
            [focalLength, 0, principalPoint.x + sensorSize.width/2],
            [0, focalLength, principalPoint.y + sensorSize.height/2],
            [0, 0, 1]
        ];

        // Calculate rotation matrix for toe-in angle
        const toeInRad = (this.params.toeIn * Math.PI) / 180;
        const R = this.calculateRotationMatrix(toeInRad);

        // Calculate translation vector (convert baseline from cm to mm for consistency with K)
        const t = [baseline * 10, 0, 0];  // cm to mm conversion

        // Calculate essential matrix
        const E = this.calculateEssentialMatrix(R, t);

        // Calculate fundamental matrix
        const F = this.calculateFundamentalMatrix(K, E);

        // Display matrices
        this.displayMatrix('calibrationMatrix', K, 'K (mm)');
        this.displayMatrix('essentialMatrix', E, 'E (mm)');
        this.displayMatrix('fundamentalMatrix', F, 'F');
    }

    calculateRotationMatrix(toeIn) {
        const cos = Math.cos(toeIn);
        const sin = Math.sin(toeIn);

        return [
            [cos, 0, sin],
            [0, 1, 0],
            [-sin, 0, cos]
        ];
    }

    calculateEssentialMatrix(R, t) {
        try {
            // Check input matrices
            if (!R || !t || !R.length || t.length !== 3) {
                throw new Error('Invalid input matrices');
            }

            // E = [t]× R
            const tx = t[0], ty = t[1], tz = t[2];
            const tCross = [
                [0, -tz, ty],
                [tz, 0, -tx],
                [-ty, tx, 0]
            ];

            const E = this.multiplyMatrices(tCross, R);

            // Normalize E to ensure numerical stability
            const norm = Math.sqrt(E.flat().reduce((sum, val) => sum + val * val, 0));
            if (norm > 1e-10) {
                return E.map(row => row.map(val => val / norm));
            }
            return E;
        } catch (error) {
            console.error('Error calculating essential matrix:', error);
            return [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ];
        }
    }

    calculateFundamentalMatrix(K, E) {
        try {
            // Check input matrices
            if (!K || !E || !K.length || !E.length || 
                K.length !== 3 || E.length !== 3) {
                throw new Error('Invalid input matrices');
            }

            // F = K^(-T) E K^(-1)
            const Kinv = this.invertMatrix(K);
            const KinvT = this.transposeMatrix(Kinv);
            const F = this.multiplyMatrices(
                KinvT,
                this.multiplyMatrices(E, Kinv)
            );

            // Normalize F to ensure numerical stability
            const norm = Math.sqrt(F.flat().reduce((sum, val) => sum + val * val, 0));
            if (norm > 1e-10) {
                return F.map(row => row.map(val => val / norm));
            }
            return F;
        } catch (error) {
            console.error('Error calculating fundamental matrix:', error);
            return [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ];
        }
    }

    displayMatrix(elementId, matrix, name) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const matrixContent = element.querySelector('.matrix-content');
        if (!matrixContent) return;

        matrixContent.innerHTML = this.formatMatrix(matrix);

        // Trigger MathJax to reprocess
        if (window.MathJax) {
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, matrixContent]);
        }
    }

    formatMatrix(matrix) {
        return '\\[\\begin{bmatrix} ' +
            matrix.map(row =>
                row.map(val => Number(val).toFixed(3)).join(' & ')
            ).join(' \\\\ ') +
            ' \\end{bmatrix}\\]';
    }

    multiplyMatrices(a, b) {
        try {
            // Check matrix dimensions
            if (!a || !b || !a.length || !b.length || !b[0] || 
                a[0].length !== b.length) {
                console.error('Invalid matrix dimensions for multiplication');
                return [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ];
            }

            const result = Array(a.length).fill().map(() => Array(b[0].length).fill(0));
            for (let i = 0; i < a.length; i++) {
                for (let j = 0; j < b[0].length; j++) {
                    for (let k = 0; k < b.length; k++) {
                        const val = a[i][k] * b[k][j];
                        // Check for NaN or Infinity
                        if (!isFinite(val)) {
                            console.warn('Non-finite value in matrix multiplication');
                            result[i][j] = 0;
                        } else {
                            result[i][j] += val;
                        }
                    }
                }
            }
            return result;
        } catch (error) {
            console.error('Error in matrix multiplication:', error);
            return [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ];
        }
    }

    invertMatrix(m) {
        try {
            // Simple 3x3 matrix inversion
            const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                     - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                     + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

            // Check for singular matrix
            if (Math.abs(det) < 1e-10) {
                console.warn('Matrix is singular or nearly singular');
                return [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ];
            }

            const invDet = 1 / det;
            
            return [
                [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet,
                 (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet,
                 (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet],
                [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet,
                 (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet,
                 (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet],
                [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet,
                 (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet,
                 (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet]
            ];
        } catch (error) {
            console.error('Error in matrix inversion:', error);
            return [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ];
        }
    }

    transposeMatrix(m) {
        return m[0].map((_, i) => m.map(row => row[i]));
    }

    setupEventListeners() {
        // Add input event listeners for all controls
        const inputs = {
            focalLength: 'focalLengthValue',
            baseline: 'baselineValue',
            pointDepth: 'pointDepthValue',
            toeIn: 'toeInValue',
            principalX: 'principalXValue',
            principalY: 'principalYValue'
        };

        Object.entries(inputs).forEach(([id, valueId]) => {
            const input = document.getElementById(id);
            const value = document.getElementById(valueId);
            if (input && value) {
                input.addEventListener('input', (e) => {
                    if (id === 'principalX' || id === 'principalY') {
                        this.params.principalPoint[id === 'principalX' ? 'x' : 'y'] = parseFloat(e.target.value);
                    } else {
                        this.params[id] = parseFloat(e.target.value);
                    }
                    value.textContent = e.target.value;
                    this.updateScene();
                });
            }
        });

        // Add regenerate features button handler
        const regenerateBtn = document.getElementById('regenerateFeatures');
        if (regenerateBtn) {
            regenerateBtn.addEventListener('click', () => {
                this.regenerateFeatures();
            });
        }

        // Add view control listeners
        const views = {
            topView: this.views.top,
            frontView: this.views.front,
            isometricView: this.views.isometric
        };

        Object.entries(views).forEach(([id, position]) => {
            const button = document.getElementById(id);
            if (button) {
                button.addEventListener('click', () => {
                    // Remove active class from all buttons
                    Object.keys(views).forEach(viewId => {
                        document.getElementById(viewId).classList.remove('active');
                    });
                    
                    // Add active class to clicked button
                    button.classList.add('active');

                    // Animate camera to new position
                    this.animateCamera(position);
                });
            }
        });

        // Add window resize handler
        window.addEventListener('resize', () => {
            const container = document.getElementById('three-container');
            this.camera.aspect = container.clientWidth / container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(container.clientWidth, container.clientHeight);
        });

        // Update baseline slider for centimeter units
        const baselineSlider = document.getElementById('baseline');
        if (baselineSlider) {
            baselineSlider.min = this.params.minBaseline;  // 10 cm
            baselineSlider.max = 200;  // 200 cm (2 meters)
            baselineSlider.step = 1;   // 1cm steps
            baselineSlider.value = 50;  // Set default to 50 cm
            this.params.baseline = 50;  // Ensure param is synced
            document.getElementById('baselineValue').textContent = '50 cm';  // Update display
            baselineSlider.addEventListener('input', (e) => {
                this.params.baseline = Math.max(this.params.minBaseline, parseFloat(e.target.value));
                document.getElementById('baselineValue').textContent = `${this.params.baseline} cm`;
                this.updateScene();
            });
        }

        // Update point depth slider for centimeter units
        const depthSlider = document.getElementById('pointDepth');
        if (depthSlider) {
            depthSlider.min = 100;   // 1m in cm
            depthSlider.max = 2000;  // 20m in cm
            depthSlider.step = 50;   // 50cm steps
            depthSlider.value = this.params.pointDepth;
            depthSlider.addEventListener('input', (e) => {
                this.params.pointDepth = parseFloat(e.target.value);
                document.getElementById('pointDepthValue').textContent = `${this.params.pointDepth} cm`;
                this.updateScene();
            });
        }
    }

    animateCamera(targetPosition, duration = 1000) {
        const startPosition = this.camera.position.clone();
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Use smooth easing function
            const eased = this.easeInOutCubic(progress);

            // Interpolate camera position
            this.camera.position.lerpVectors(startPosition, targetPosition, eased);
            this.camera.lookAt(0, 0, 0);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    easeInOutCubic(x) {
        return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        
        // Update camera views each frame
        this.updateCameraViews();
    }

    createRandomFeatures(count) {
        const featureGeometry = new THREE.SphereGeometry(0.05);
        const featureMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
        
        for (let i = 0; i < count; i++) {
            const feature = new THREE.Mesh(featureGeometry, featureMaterial);
            // Random position in a reasonable volume
            feature.position.set(
                (Math.random() - 0.5) * 4,
                (Math.random() - 0.5) * 2,
                -(Math.random() * 4 + 2)
            );
            this.features.push(feature);
            this.scene.add(feature);
        }
    }

    createProjectionPlanes() {
        // Create separate canvases for camera views
        const planeGeometry = new THREE.PlaneGeometry(0.4, 0.3);
        const planeMaterial = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide
        });

        // Left camera projection
        const leftCanvas = document.createElement('canvas');
        leftCanvas.width = 400;
        leftCanvas.height = 300;
        this.projectionPlanes.left = {
            mesh: new THREE.Mesh(planeGeometry, planeMaterial.clone()),
            canvas: leftCanvas,
            camera: new THREE.PerspectiveCamera(60, 4/3, 0.1, 1000),
            scene: new THREE.Scene()
        };
        this.projectionPlanes.left.scene.background = new THREE.Color(0xf0f0f0);
        this.projectionPlanes.left.mesh.position.z = 0.6;
        this.camera1.add(this.projectionPlanes.left.mesh);

        // Right camera projection
        const rightCanvas = document.createElement('canvas');
        rightCanvas.width = 400;
        rightCanvas.height = 300;
        this.projectionPlanes.right = {
            mesh: new THREE.Mesh(planeGeometry, planeMaterial.clone()),
            canvas: rightCanvas,
            camera: new THREE.PerspectiveCamera(60, 4/3, 0.1, 1000),
            scene: new THREE.Scene()
        };
        this.projectionPlanes.right.scene.background = new THREE.Color(0xf0f0f0);
        this.projectionPlanes.right.mesh.position.z = 0.6;
        this.camera2.add(this.projectionPlanes.right.mesh);

        // Set up canvases and mount them to the DOM
        const leftView = document.getElementById('leftView');
        const rightView = document.getElementById('rightView');

        if (!leftView || !rightView) {
            console.error('Could not find view containers');
            return;
        }

        Object.entries(this.projectionPlanes).forEach(([side, plane]) => {
            // Create texture from canvas
            const texture = new THREE.CanvasTexture(plane.canvas);
            plane.mesh.material.map = texture;
            plane.mesh.material.needsUpdate = true;
            
            // Add lights to camera scenes
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            plane.scene.add(ambientLight);
            plane.scene.add(directionalLight);

            // Add grid helper to camera scenes
            const gridHelper = new THREE.GridHelper(10, 10);
            plane.scene.add(gridHelper);

            // Mount canvases to DOM
            const container = side === 'left' ? leftView : rightView;
            container.appendChild(plane.canvas);
        });
    }

    updateCameraViews() {
        try {
            Object.entries(this.projectionPlanes).forEach(([side, plane]) => {
                // Store references to important objects
                const lights = plane.scene.children.filter(child => 
                    child instanceof THREE.AmbientLight || 
                    child instanceof THREE.DirectionalLight
                );
                const grid = plane.scene.children.find(child => 
                    child instanceof THREE.GridHelper
                );

                // Clear previous features from scene, keeping lights and grid
                plane.scene.children.forEach(child => {
                    if (!lights.includes(child) && child !== grid) {
                        plane.scene.remove(child);
                        if (child.geometry) child.geometry.dispose();
                        if (child.material) child.material.dispose();
                    }
                });

                // Add lights and grid if they don't exist
                if (lights.length === 0) {
                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                    directionalLight.position.set(5, 5, 5);
                    plane.scene.add(ambientLight);
                    plane.scene.add(directionalLight);
                }
                if (!grid) {
                    plane.scene.add(new THREE.GridHelper(10, 10));
                }

                // Add features to camera scene
                this.features.forEach((feature, index) => {
                    const featureClone = feature.clone();
                    featureClone.material = feature.material.clone();
                    plane.scene.add(featureClone);
                });

                // Add convergence point to scene
                const convergenceClone = this.convergencePoint.clone();
                convergenceClone.material = this.convergencePoint.material.clone();
                plane.scene.add(convergenceClone);

                // Update camera parameters
                const { focalLength, sensorSize } = this.params;
                const aspect = sensorSize.width / sensorSize.height;
                const fov = 2 * Math.atan(sensorSize.height / (2 * focalLength)) * (180 / Math.PI);
                
                plane.camera.fov = fov;
                plane.camera.aspect = aspect;
                plane.camera.updateProjectionMatrix();

                // Position camera for rendering
                if (side === 'left') {
                    plane.camera.position.copy(this.camera1.position);
                    plane.camera.rotation.copy(this.camera1.rotation);
                } else {
                    plane.camera.position.copy(this.camera2.position);
                    plane.camera.rotation.copy(this.camera2.rotation);
                }

                // Render the scene
                plane.renderer.render(plane.scene, plane.camera);
                plane.mesh.material.map.needsUpdate = true;
            });
        } catch (error) {
            console.error('Error in updateCameraViews:', error);
        }
    }

    regenerateFeatures() {
        // Remove existing features
        this.features.forEach(feature => {
            this.scene.remove(feature);
        });
        this.features = [];

        // Create new random features
        this.createRandomFeatures(10);
        this.updateScene();
    }
}

// Initialize when the page loads
window.addEventListener('load', () => {
    new MultiviewGeometry();
}); 