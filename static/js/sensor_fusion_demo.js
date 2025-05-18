// Initialize D3.js visualization
const svg = d3.select('#sensorFusionSVG');
const width = +svg.attr('width');
const height = +svg.attr('height');

// Create path generators
const truePath = d3.line();
const gpsPath = d3.line();
const imuPath = d3.line();
const kalmanPath = d3.line();

// Initialize data arrays
let truePositions = [];
let gpsReadings = [];
let imuReadings = [];
let kalmanEstimates = [];
let uncertaintyEllipses = [];

// Initialize Kalman filter
const kf = new KalmanFilter();

// Animation parameters
let animationId = null;
let showPaths = true;
let showUncertainty = true;

// Initialize noise parameters
let gpsNoise = 30;
let imuNoise = 10;

// True trajectory parameters
const center = { x: width/2, y: height/2 };
const radius = Math.min(width, height) * 0.3;
let angle = 0;

// Error metrics
let gpsError = 0;
let kalmanError = 0;

// Update noise values from sliders
document.getElementById('gpsNoise').addEventListener('input', function(e) {
    gpsNoise = +e.target.value;
    document.getElementById('gpsNoiseValue').textContent = gpsNoise;
});

document.getElementById('imuNoise').addEventListener('input', function(e) {
    imuNoise = +e.target.value;
    document.getElementById('imuNoiseValue').textContent = imuNoise;
});

// Button event listeners
document.getElementById('startFusion').addEventListener('click', toggleSimulation);
document.getElementById('resetFusion').addEventListener('click', resetSimulation);
document.getElementById('togglePath').addEventListener('click', togglePaths);
document.getElementById('toggleUncertainty').addEventListener('click', () => {
    showUncertainty = !showUncertainty;
    draw();
});

function toggleSimulation() {
    const button = document.getElementById('startFusion');
    if (animationId) {
        button.textContent = 'Start';
        cancelAnimationFrame(animationId);
        animationId = null;
    } else {
        button.textContent = 'Stop';
        animate();
    }
}

function resetSimulation() {
    // Clear all data arrays
    truePositions = [];
    gpsReadings = [];
    imuReadings = [];
    kalmanEstimates = [];
    uncertaintyEllipses = [];
    
    // Reset angle
    angle = 0;
    
    // Reset Kalman filter with initial state at center
    kf.state = [center.x, center.y, 0, 0];
    
    // Reset error metrics
    gpsError = 0;
    kalmanError = 0;
    updateErrorMetrics();
    
    // Redraw
    draw();
}

function togglePaths() {
    showPaths = !showPaths;
    draw();
}

function addGaussianNoise(value, std) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return value + z * std;
}

function calculateRMSE(predictions, truths) {
    if (predictions.length === 0) return 0;
    const squaredErrors = predictions.map((pred, i) => {
        const dx = pred[0] - truths[i][0];
        const dy = pred[1] - truths[i][1];
        return dx * dx + dy * dy;
    });
    return Math.sqrt(squaredErrors.reduce((a, b) => a + b) / predictions.length);
}

function updateErrorMetrics() {
    document.getElementById('gpsError').textContent = gpsError.toFixed(2);
    document.getElementById('kalmanError').textContent = kalmanError.toFixed(2);
}

function drawUncertaintyEllipse(svg, x, y, sigmaX, sigmaY, color) {
    // Scale factor for 95% confidence (2-sigma)
    const scale = 2;
    
    // Create ellipse path
    const ellipse = svg.append('ellipse')
        .attr('cx', x)
        .attr('cy', y)
        .attr('rx', sigmaX * scale)
        .attr('ry', sigmaY * scale)
        .attr('fill', color)
        .attr('fill-opacity', 0.1)
        .attr('stroke', color)
        .attr('stroke-opacity', 0.3);
}

function animate() {
    // Calculate true position (circular motion)
    const trueX = center.x + radius * Math.cos(angle);
    const trueY = center.y + radius * Math.sin(angle);
    truePositions.push([trueX, trueY]);

    // Generate noisy GPS reading
    const gpsX = addGaussianNoise(trueX, gpsNoise);
    const gpsY = addGaussianNoise(trueY, gpsNoise);
    gpsReadings.push([gpsX, gpsY]);

    // Generate noisy IMU reading (velocity)
    const dt = 0.1;  // Time step
    const trueVx = -radius * Math.sin(angle);  // True velocity
    const trueVy = radius * Math.cos(angle);
    
    // Add noise proportional to velocity magnitude
    const velMagnitude = Math.sqrt(trueVx * trueVx + trueVy * trueVy);
    const imuVx = addGaussianNoise(trueVx, imuNoise * 0.1 + velMagnitude * 0.05);
    const imuVy = addGaussianNoise(trueVy, imuNoise * 0.1 + velMagnitude * 0.05);
    
    // Better IMU integration using trapezoidal rule
    const lastImuPos = imuReadings.length > 0 ? imuReadings[imuReadings.length - 1] : [center.x, center.y];
    const lastVx = imuReadings.length > 0 ? (lastImuPos[0] - imuReadings[Math.max(0, imuReadings.length - 2)][0]) / dt : 0;
    const lastVy = imuReadings.length > 0 ? (lastImuPos[1] - imuReadings[Math.max(0, imuReadings.length - 2)][1]) / dt : 0;
    
    imuReadings.push([
        lastImuPos[0] + dt * (imuVx + lastVx) / 2,  // Trapezoidal integration
        lastImuPos[1] + dt * (imuVy + lastVy) / 2
    ]);

    // Process measurement matrix (GPS noise)
    const R = [
        [gpsNoise * gpsNoise, 0],
        [0, gpsNoise * gpsNoise]
    ];
    
    // Process noise matrix (IMU noise + system uncertainty)
    const dt2 = dt * dt;
    const baseProcessNoise = 0.1;  // Base process noise
    const velocityNoise = imuNoise * 0.1 + velMagnitude * 0.05;  // Velocity-dependent noise
    
    const Q = [
        [dt2 * baseProcessNoise, 0, dt * baseProcessNoise, 0],
        [0, dt2 * baseProcessNoise, 0, dt * baseProcessNoise],
        [dt * baseProcessNoise, 0, velocityNoise, 0],
        [0, dt * baseProcessNoise, 0, velocityNoise]
    ];

    try {
        // Kalman filter prediction with velocity control input
        const u = [0, 0, imuVx, imuVy];  // Use IMU velocity as control input
        kf.predict(Q, u);
        
        // Update with GPS measurement
        kf.update([gpsX, gpsY], R);
        
        const state = kf.getState();
        kalmanEstimates.push(state.position);
        uncertaintyEllipses.push({
            x: state.position[0],
            y: state.position[1],
            sigmaX: state.uncertainty.position[0],
            sigmaY: state.uncertainty.position[1]
        });

        // Update error metrics
        gpsError = calculateRMSE(gpsReadings, truePositions);
        kalmanError = calculateRMSE(kalmanEstimates, truePositions);
        updateErrorMetrics();
    } catch (error) {
        console.error('Kalman filter error:', error);
    }

    // Draw updated visualization
    draw();

    // Update angle for next frame
    angle += 0.02;
    
    // Request next frame
    animationId = requestAnimationFrame(animate);
}

function draw() {
    // Clear SVG
    svg.selectAll('*').remove();

    if (showPaths) {
        // Draw true path
        svg.append('path')
            .datum(truePositions)
            .attr('d', truePath)
            .attr('fill', 'none')
            .attr('stroke', 'black')
            .attr('stroke-width', 2);

        // Draw GPS readings path
        svg.append('path')
            .datum(gpsReadings)
            .attr('d', gpsPath)
            .attr('fill', 'none')
            .attr('stroke', 'red')
            .attr('stroke-width', 1)
            .style('opacity', 0.5);

        // Draw IMU path
        svg.append('path')
            .datum(imuReadings)
            .attr('d', imuPath)
            .attr('fill', 'none')
            .attr('stroke', 'green')
            .attr('stroke-width', 1)
            .style('opacity', 0.5);

        // Draw Kalman filter path
        svg.append('path')
            .datum(kalmanEstimates)
            .attr('d', kalmanPath)
            .attr('fill', 'none')
            .attr('stroke', 'blue')
            .attr('stroke-width', 2);
    }

    // Draw uncertainty ellipses
    if (showUncertainty && uncertaintyEllipses.length > 0) {
        const lastEllipse = uncertaintyEllipses[uncertaintyEllipses.length - 1];
        drawUncertaintyEllipse(
            svg,
            lastEllipse.x,
            lastEllipse.y,
            lastEllipse.sigmaX,
            lastEllipse.sigmaY,
            'blue'
        );
    }

    // Draw current positions
    if (truePositions.length > 0) {
        const lastTrue = truePositions[truePositions.length - 1];
        const lastGPS = gpsReadings[gpsReadings.length - 1];
        const lastIMU = imuReadings[imuReadings.length - 1];
        const lastKalman = kalmanEstimates[kalmanEstimates.length - 1];

        // True position
        svg.append('circle')
            .attr('cx', lastTrue[0])
            .attr('cy', lastTrue[1])
            .attr('r', 5)
            .attr('fill', 'black');

        // GPS reading
        svg.append('circle')
            .attr('cx', lastGPS[0])
            .attr('cy', lastGPS[1])
            .attr('r', 5)
            .attr('fill', 'red');

        // IMU position
        svg.append('circle')
            .attr('cx', lastIMU[0])
            .attr('cy', lastIMU[1])
            .attr('r', 5)
            .attr('fill', 'green');

        // Kalman estimate
        svg.append('circle')
            .attr('cx', lastKalman[0])
            .attr('cy', lastKalman[1])
            .attr('r', 5)
            .attr('fill', 'blue');
    }
} 