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
        this.epipolarPlane = null;
        this.epipolarLines = null;
        this.gridHelper = null;
        
        this.params = {
            focalLength: 35,
            baseline: 0.5,
            pointDepth: 10,
            toeIn: 0,
            principalPoint: { x: 0, y: 0 },
            sensorSize: { width: 36, height: 24 } // 35mm format
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

        // Create epipolar plane
        const planeGeometry = new THREE.PlaneGeometry(5, 5);
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });
        this.epipolarPlane = new THREE.Mesh(planeGeometry, planeMaterial);
        this.scene.add(this.epipolarPlane);

        // Create epipolar lines
        const lineGeometry = new THREE.BufferGeometry();
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff });
        this.epipolarLines = new THREE.LineSegments(lineGeometry, lineMaterial);
        this.scene.add(this.epipolarLines);

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
        const { focalLength, baseline, pointDepth, toeIn } = this.params;

        // Update camera positions (baseline now in meters)
        this.camera1.position.set(-baseline/2, 0, 0);
        this.camera2.position.set(baseline/2, 0, 0);

        // Update camera rotations using toe-in angle
        const toeInRad = (toeIn * Math.PI) / 180;
        this.camera1.rotation.set(0, toeInRad, 0);
        this.camera2.rotation.set(0, -toeInRad, 0);

        // Update convergence point
        if (toeInRad !== 0) {
            const convergenceDistance = (baseline/2) / Math.tan(Math.abs(toeInRad));
            this.convergencePoint.position.set(0, 0, -convergenceDistance);
        } else {
            this.convergencePoint.position.set(0, 0, -pointDepth);
        }

        // Update feature projections and epipolar lines
        this.updateFeatureProjections();
        this.updateEpipolarGeometry();
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
                // Convert world coordinates to camera coordinates
                const featurePos = feature.position.clone();
                
                // Project to left camera
                const leftProj = this.projectToCamera(featurePos, this.camera1, this.projectionPlanes.left);
                if (leftProj) this.featureProjections.left.push(leftProj);

                // Project to right camera
                const rightProj = this.projectToCamera(featurePos, this.camera2, this.projectionPlanes.right);
                if (rightProj) this.featureProjections.right.push(rightProj);
            });

            // Update projection plane textures
            this.updateProjectionPlaneTextures();
        } catch (error) {
            console.error('Error in updateFeatureProjections:', error);
        }
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

            // Check if point is in front of camera
            if (pointInCameraSpace.z > 0) return null;

            // Project to image plane with principal point offset
            const { focalLength, principalPoint, sensorSize } = this.params;
            const x = -(pointInCameraSpace.x * focalLength) / pointInCameraSpace.z + principalPoint.x;
            const y = -(pointInCameraSpace.y * focalLength) / pointInCameraSpace.z + principalPoint.y;

            // Check if projection is within sensor bounds
            if (Math.abs(x) > sensorSize.width/2 || Math.abs(y) > sensorSize.height/2) {
                return null;
            }

            return { x, y, z: pointInCameraSpace.z };
        } catch (error) {
            console.error('Error in projectToCamera:', error);
            return null;
        }
    }

    updateProjectionPlaneTextures() {
        try {
            Object.entries(this.projectionPlanes).forEach(([side, plane]) => {
                if (!plane || !plane.renderer || !plane.renderer.domElement) {
                    console.warn(`Missing projection plane elements for ${side} camera`);
                    return;
                }

                const ctx = plane.renderer.domElement.getContext('2d');
                if (!ctx) {
                    console.warn(`Could not get 2D context for ${side} camera`);
                    return;
                }

                ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

                // Draw grid lines
                ctx.strokeStyle = '#cccccc';
                ctx.lineWidth = 1;
                const gridSize = 20;
                for (let i = 0; i <= ctx.canvas.width; i += gridSize) {
                    ctx.beginPath();
                    ctx.moveTo(i, 0);
                    ctx.lineTo(i, ctx.canvas.height);
                    ctx.stroke();
                }
                for (let i = 0; i <= ctx.canvas.height; i += gridSize) {
                    ctx.beginPath();
                    ctx.moveTo(0, i);
                    ctx.lineTo(ctx.canvas.width, i);
                    ctx.stroke();
                }

                // Draw principal point
                const { principalPoint, sensorSize } = this.params;
                const ppX = (principalPoint.x / sensorSize.width + 0.5) * ctx.canvas.width;
                const ppY = (principalPoint.y / sensorSize.height + 0.5) * ctx.canvas.height;
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 2;
                const crossSize = 10;
                ctx.beginPath();
                ctx.moveTo(ppX - crossSize, ppY);
                ctx.lineTo(ppX + crossSize, ppY);
                ctx.moveTo(ppX, ppY - crossSize);
                ctx.lineTo(ppX, ppY + crossSize);
                ctx.stroke();

                // Draw epipolar lines and feature correspondences
                const otherSide = side === 'left' ? 'right' : 'left';
                const otherCamera = side === 'left' ? this.camera2 : this.camera1;
                const thisCamera = side === 'left' ? this.camera1 : this.camera2;
                
                // Calculate fundamental matrix once for all features
                const F = this.calculateFundamentalMatrix(
                    this.getCalibrationMatrix(),
                    this.calculateEssentialMatrix(
                        this.calculateRotationMatrix(this.params.toeIn * Math.PI / 180),
                        [this.params.baseline/100, 0, 0]
                    )
                );

                // Store current feature projections for correspondence lines
                const currentProjections = [];
                
                this.features.forEach((feature, index) => {
                    const featurePos = feature.position;
                    const proj = this.projectToCamera(featurePos, thisCamera, plane);
                    
                    if (proj) {
                        // Convert to normalized image coordinates
                        const x = (proj.x / sensorSize.width + 0.5) * ctx.canvas.width;
                        const y = (proj.y / sensorSize.height + 0.5) * ctx.canvas.height;
                        currentProjections.push({ x, y, index });

                        // Draw feature point
                        ctx.fillStyle = `hsl(${(index * 30) % 360}, 100%, 50%)`;
                        ctx.beginPath();
                        ctx.arc(x, y, 4, 0, Math.PI * 2);
                        ctx.fill();

                        // Draw feature ID
                        ctx.fillStyle = '#000000';
                        ctx.font = '12px Arial';
                        ctx.fillText(index + 1, x + 5, y - 5);

                        // Calculate and draw epipolar line in other view
                        const point = [x, y, 1];
                        const line = side === 'left' ? 
                            this.multiplyMatrixVector(F, point) :
                            this.multiplyMatrixVector(this.transposeMatrix(F), point);

                        // Draw epipolar line
                        const [a, b, c] = line;
                        ctx.strokeStyle = `hsla(${(index * 30) % 360}, 100%, 50%, 0.3)`;
                        ctx.lineWidth = 1;
                        
                        const intersections = this.findLineImageIntersections(a, b, c, ctx.canvas.width, ctx.canvas.height);
                        if (intersections.length === 2) {
                            ctx.beginPath();
                            ctx.moveTo(intersections[0].x, intersections[0].y);
                            ctx.lineTo(intersections[1].x, intersections[1].y);
                            ctx.stroke();
                        }
                    }
                });

                // Draw correspondence lines between views
                if (side === 'right') {
                    const leftProjections = this.featureProjections.left;
                    const rightProjections = this.featureProjections.right;
                    
                    if (leftProjections && rightProjections) {
                        leftProjections.forEach((leftProj, index) => {
                            const rightProj = rightProjections[index];
                            if (leftProj && rightProj) {
                                const leftX = (leftProj.x / sensorSize.width + 0.5) * ctx.canvas.width;
                                const rightX = (rightProj.x / sensorSize.width + 0.5) * ctx.canvas.width;
                                const leftY = (leftProj.y / sensorSize.height + 0.5) * ctx.canvas.height;
                                const rightY = (rightProj.y / sensorSize.height + 0.5) * ctx.canvas.height;

                                // Draw correspondence line
                                const leftView = document.getElementById('leftView');
                                const rightView = document.getElementById('rightView');
                                if (leftView && rightView) {
                                    const leftRect = leftView.getBoundingClientRect();
                                    const rightRect = rightView.getBoundingClientRect();
                                    
                                    // Create or update correspondence line
                                    const lineId = `correspondence-${index}`;
                                    let line = document.getElementById(lineId);
                                    if (!line) {
                                        line = document.createElement('div');
                                        line.id = lineId;
                                        line.style.position = 'absolute';
                                        line.style.pointerEvents = 'none';
                                        line.style.zIndex = '1000';
                                        document.body.appendChild(line);
                                    }

                                    // Calculate line position and angle
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
                            }
                        });
                    }
                }

                // Update texture
                if (plane.mesh && plane.mesh.material && plane.mesh.material.map) {
                    plane.mesh.material.map.needsUpdate = true;
                }
            });
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

    updateEpipolarGeometry() {
        const point = this.point3D.position;
        const cam1Pos = this.camera1.position;
        const cam2Pos = this.camera2.position;

        // Update epipolar plane
        const normal = new THREE.Vector3().crossVectors(
            new THREE.Vector3().subVectors(point, cam1Pos),
            new THREE.Vector3().subVectors(point, cam2Pos)
        ).normalize();

        this.epipolarPlane.lookAt(normal);
        this.epipolarPlane.position.copy(point);

        // Update epipolar lines
        const linePoints = new Float32Array([
            cam1Pos.x, cam1Pos.y, cam1Pos.z,
            point.x, point.y, point.z,
            cam2Pos.x, cam2Pos.y, cam2Pos.z
        ]);

        this.epipolarLines.geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(linePoints, 3)
        );

        // Add epipolar lines for each feature
        this.features.forEach(feature => {
            const featurePos = feature.position;
            const cam1Pos = new THREE.Vector3();
            const cam2Pos = new THREE.Vector3();
            this.camera1.getWorldPosition(cam1Pos);
            this.camera2.getWorldPosition(cam2Pos);

            // Create epipolar line geometry
            const lineGeometry = new THREE.BufferGeometry();
            const linePoints = new Float32Array([
                cam1Pos.x, cam1Pos.y, cam1Pos.z,
                featurePos.x, featurePos.y, featurePos.z,
                cam2Pos.x, cam2Pos.y, cam2Pos.z
            ]);
            lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePoints, 3));
            
            // Create line material
            const lineMaterial = new THREE.LineBasicMaterial({ 
                color: 0x0000ff,
                transparent: true,
                opacity: 0.3
            });

            // Add line to scene
            const line = new THREE.Line(lineGeometry, lineMaterial);
            this.scene.add(line);
        });
    }

    updateMatrices() {
        const { focalLength, baseline, principalPoint, sensorSize } = this.params;

        // Calculate calibration matrix with principal point
        const K = [
            [focalLength, 0, principalPoint.x + sensorSize.width/2],
            [0, focalLength, principalPoint.y + sensorSize.height/2],
            [0, 0, 1]
        ];

        // Calculate rotation matrix for toe-in angle
        const toeInRad = (this.params.toeIn * Math.PI) / 180;
        const R = this.calculateRotationMatrix(toeInRad);

        // Calculate translation vector
        const t = [baseline/100, 0, 0]; // Convert cm to meters

        // Calculate essential matrix
        const E = this.calculateEssentialMatrix(R, t);

        // Calculate fundamental matrix
        const F = this.calculateFundamentalMatrix(K, E);

        // Display matrices
        this.displayMatrix('calibrationMatrix', K, 'K');
        this.displayMatrix('essentialMatrix', E, 'E');
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
        // E = [t]× R
        const tx = t[0], ty = t[1], tz = t[2];
        const tCross = [
            [0, -tz, ty],
            [tz, 0, -tx],
            [-ty, tx, 0]
        ];

        return this.multiplyMatrices(tCross, R);
    }

    calculateFundamentalMatrix(K, E) {
        // F = K^(-T) E K^(-1)
        const Kinv = this.invertMatrix(K);
        const KinvT = this.transposeMatrix(Kinv);
        return this.multiplyMatrices(
            KinvT,
            this.multiplyMatrices(E, Kinv)
        );
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

    invertMatrix(m) {
        // Simple 3x3 matrix inversion
        const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                 - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                 + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

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
        // Create separate renderers for camera views
        const planeGeometry = new THREE.PlaneGeometry(0.4, 0.3);
        const planeMaterial = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide
        });

        // Left camera projection
        this.projectionPlanes.left = {
            mesh: new THREE.Mesh(planeGeometry, planeMaterial.clone()),
            renderer: new THREE.WebGLRenderer({ antialias: true }),
            camera: new THREE.PerspectiveCamera(60, 4/3, 0.1, 1000),
            scene: new THREE.Scene()
        };
        this.projectionPlanes.left.scene.background = new THREE.Color(0xf0f0f0);
        this.projectionPlanes.left.mesh.position.z = 0.6;
        this.camera1.add(this.projectionPlanes.left.mesh);

        // Right camera projection
        this.projectionPlanes.right = {
            mesh: new THREE.Mesh(planeGeometry, planeMaterial.clone()),
            renderer: new THREE.WebGLRenderer({ antialias: true }),
            camera: new THREE.PerspectiveCamera(60, 4/3, 0.1, 1000),
            scene: new THREE.Scene()
        };
        this.projectionPlanes.right.scene.background = new THREE.Color(0xf0f0f0);
        this.projectionPlanes.right.mesh.position.z = 0.6;
        this.camera2.add(this.projectionPlanes.right.mesh);

        // Set up renderers and mount them to the DOM
        const leftView = document.getElementById('leftView');
        const rightView = document.getElementById('rightView');

        Object.entries(this.projectionPlanes).forEach(([side, plane]) => {
            plane.renderer.setSize(400, 300);
            plane.mesh.material.map = new THREE.CanvasTexture(plane.renderer.domElement);
            
            // Add lights to camera scenes
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            plane.scene.add(ambientLight);
            plane.scene.add(directionalLight);

            // Add grid helper to camera scenes
            const gridHelper = new THREE.GridHelper(10, 10);
            plane.scene.add(gridHelper);

            // Mount renderers to DOM
            const container = side === 'left' ? leftView : rightView;
            container.appendChild(plane.renderer.domElement);
        });
    }

    updateCameraViews() {
        try {
            Object.entries(this.projectionPlanes).forEach(([side, plane]) => {
                // Clear previous features from scene
                while(plane.scene.children.length > 0) { 
                    plane.scene.remove(plane.scene.children[0]);
                }

                // Add lights and grid back
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 5, 5);
                plane.scene.add(ambientLight);
                plane.scene.add(directionalLight);
                plane.scene.add(new THREE.GridHelper(10, 10));

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