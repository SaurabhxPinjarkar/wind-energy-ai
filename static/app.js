// Initialize Map with dark theme
const map = L.map('map', {
    zoomControl: false // Custom position if needed later
}).setView([18.5, 75.0], 7);

// Using standard OpenStreetMap tiles for better geographic visibility
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
}).addTo(map);

// Add zoom control to top right
L.control.zoom({ position: 'topright' }).addTo(map);

let markers = [];
let allLocations = [];
let currentPrediction = null;
let comparisonList = [];

// Load model info onto the UI
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        if (data.status === 'success') {
            document.getElementById('model-name').innerText = `Model: ${data.model}`;
            document.getElementById('model-r2').innerText = data.r2_score;
            document.getElementById('model-rmse').innerText = data.rmse;
            document.getElementById('threshold').innerText = data.energy_threshold;
        }
    } catch (err) {
        console.error("Failed to load model info", err);
    }
}

// Load all pre-selected 100 locations
async function loadAllLocations() {
    try {
        const response = await fetch('/api/locations');
        const data = await response.json();
        if (data.status === 'success') {
            allLocations = data.locations;
        }
    } catch (err) {
        console.error("Failed to load locations", err);
    }
}

// Predict energy potential for coordinates
async function predictLocation(lat, lon) {
    const resultsDiv = document.getElementById('prediction-results');
    
    // Show premium Loading Spinner
    resultsDiv.innerHTML = `
        <div class="loading-wrapper">
            <div class="spinner"></div>
            <p>Fetching satellite data...</p>
        </div>
    `;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: lat, longitude: lon })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayPredictionResults(data);
            addMarker(lat, lon, data);
        } else {
            resultsDiv.innerHTML = `<div class="status-badge status-not-suitable">Error: ${data.message}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="status-badge status-not-suitable">Connection Failed</div>`;
    }
}

// Display beautiful glassy results
function displayPredictionResults(data) {
    const isGood = data.suitability.includes('Good');
    const statusClass = isGood ? 'status-good' : 'status-not-suitable';
    
    // Smooth progress bar calculation
    const energyPercent = Math.min(100, Math.max(0, data.suitability_score || 0));

    let html = `
        <div class="info-row">
            <span class="info-label">Coordinates</span>
            <span class="info-value">${data.latitude.toFixed(4)}°, ${data.longitude.toFixed(4)}°</span>
        </div>
        <div class="info-row">
            <span class="info-label">Predicted Output</span>
            <span class="info-value" style="color: var(--primary-color); font-size: 1.1em;">${data.predicted_energy} kWh</span>
        </div>
        
        <div class="weather-grid">
            <div class="weather-item">
                <div class="weather-label">Wind Speed</div>
                <div class="weather-value">${data.wind_speed} m/s</div>
            </div>
            <div class="weather-item">
                <div class="weather-label">Temp</div>
                <div class="weather-value">${data.temperature}°C</div>
            </div>
            <div class="weather-item">
                <div class="weather-label">Pressure</div>
                <div class="weather-value">${data.pressure} mb</div>
            </div>
            <div class="weather-item">
                <div class="weather-label">Humidity</div>
                <div class="weather-value">${data.humidity}%</div>
            </div>
        </div>

        <div class="status-badge ${statusClass}">
            ${data.suitability}
        </div>

        <div class="progress-container">
            <p style="font-size: 0.85em; color: var(--text-muted); margin-bottom: 2px;">Feasibility Score</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <div class="progress-labels">
                <span>0%</span>
                <span style="color: var(--primary-color); font-weight: 600;">${energyPercent.toFixed(1)}%</span>
                <span>100%</span>
            </div>
        </div>
        
        <button class="btn-secondary" style="margin-top: 15px; width: 100%;" onclick="addToCompare()">
            Add to Compare
        </button>
    `;

    currentPrediction = data;
    const containerDiv = document.getElementById('prediction-results');
    // For smooth CSS transition rendering, fade out, mount, fade in
    containerDiv.style.opacity = '0';
    
    setTimeout(() => {
        containerDiv.innerHTML = html;
        containerDiv.style.transition = 'opacity 0.3s ease';
        containerDiv.style.opacity = '1';
        
        // Trigger CSS animation for width
        setTimeout(() => {
            const fill = containerDiv.querySelector('.progress-fill');
            if(fill) fill.style.width = `${energyPercent}%`;
        }, 100);
    }, 150);
}

// ----------------- Comparison Logic -----------------
function addToCompare() {
    if (!currentPrediction) return;
    if (comparisonList.length >= 4) {
        alert("Maximum 4 locations allowed for comparison.");
        return;
    }
    
    // duplicate check
    if (comparisonList.some(item => item.latitude === currentPrediction.latitude && item.longitude === currentPrediction.longitude)) {
        alert("Location already in comparison board.");
        return;
    }
    
    comparisonList.push(currentPrediction);
    renderComparisonBoard();
}

function clearComparison() {
    comparisonList = [];
    renderComparisonBoard();
}

function removeCompare(index) {
    comparisonList.splice(index, 1);
    renderComparisonBoard();
}

function renderComparisonBoard() {
    const grid = document.getElementById('comparison-grid');
    if (!grid) return;
    
    if (comparisonList.length === 0) {
        grid.innerHTML = `<div style="grid-column: span 4; text-align: center; color: var(--text-muted); align-self: center; font-size: 0.9em;">Analyze a location and click "Add to Compare" to build a cross-site analysis.</div>`;
        return;
    }
    
    let html = '';
    comparisonList.forEach((data, index) => {
        const isGood = data.suitability.includes('Good');
        const borderColor = isGood ? 'var(--success)' : 'var(--danger)';
        const energyPercent = Math.min(100, Math.max(0, data.suitability_score || 0));
        
        html += `
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-top: 3px solid ${borderColor}; padding: 15px; border-radius: 8px; position: relative; display: flex; flex-direction: column; gap: 8px;">
                <button onclick="removeCompare(${index})" style="position: absolute; top: 10px; right: 10px; background: transparent; border: none; color: var(--text-muted); cursor: pointer; padding: 0; width: auto; font-size: 1.2em;">&times;</button>
                <div style="font-size: 0.85em; color: var(--text-muted); margin-right: 15px;">Site ${index + 1} (${data.latitude.toFixed(2)}°, ${data.longitude.toFixed(2)}°)</div>
                <div style="font-size: 1.3em; font-weight: 600; color: ${borderColor};">${data.predicted_energy} kWh</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>Wind:</span> <span style="color: white;">${data.wind_speed} m/s</span></div>
                 <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>Temp:</span> <span style="color: white;">${data.temperature} &deg;C</span></div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85em;">
                    <span>Score:</span> 
                    <span style="color: ${borderColor}; font-weight: bold;">${energyPercent.toFixed(1)}%</span>
                </div>
            </div>
        `;
    });
    grid.innerHTML = html;
}

// Set custom markers
function addMarker(lat, lon, data) {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];

    const isGood = data.suitability.includes('Good');
    const color = isGood ? 'green' : 'red';
    
    const icon = L.divIcon({
        className: 'custom-div-icon',
        html: `<div style="background-color: ${isGood ? '#10b981' : '#ef4444'}; width: 15px; height: 15px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px ${isGood ? '#10b981' : '#ef4444'};"></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });

    const marker = L.marker([lat, lon], { icon: icon }).addTo(map);
    
    const popupContent = `
        <div class="popup-title">Analysis Result</div>
        <div class="popup-info">
            <strong>Energy:</strong> <span style="color: white">${data.predicted_energy} kWh</span><br>
            <strong>Wind:</strong> <span style="color: white">${data.wind_speed} m/s</span><br>
            <div style="margin-top: 5px; color: ${isGood ? '#10b981' : '#ef4444'}; font-weight: bold;">
                ${data.suitability}
            </div>
        </div>
    `;
    
    marker.bindPopup(popupContent).openPopup();
    markers.push(marker);
    map.setView([lat, lon], 9, { animate: true, duration: 1 });
}

function showAllLocations() {
    if (allLocations.length === 0) {
        alert('Data is still loading. Please try again in a moment.');
        return;
    }

    clearMap();

    allLocations.forEach(loc => {
        const marker = L.circleMarker([loc.latitude, loc.longitude], {
            radius: 4,
            fillColor: 'var(--primary-color)',
            color: '#fff',
            weight: 1,
            opacity: 0.8,
            fillOpacity: 0.6
        }).addTo(map);

        marker.bindPopup(`
            <div class="popup-title">${loc.place}</div>
            <div class="popup-info">Click to analyze location</div>
        `);
        
        marker.on('click', () => {
            predictLocation(loc.latitude, loc.longitude);
        });
        
        markers.push(marker);
    });

    const res = document.getElementById('prediction-results');
    res.innerHTML = `<div class="status-badge status-good" style="background: transparent; color: var(--primary-color); border: 1px solid var(--primary-color);">Loaded ${allLocations.length} Known Locations</div>`;
}

function manualPredict() {
    const lat = parseFloat(document.getElementById('lat-input').value);
    const lon = parseFloat(document.getElementById('lon-input').value);

    if (isNaN(lat) || isNaN(lon)) return alert('Enter valid coordinates');
    predictLocation(lat, lon);
}

function clearMap() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    document.getElementById('prediction-results').innerHTML = 
        '<p style="color: var(--text-muted); text-align: center; margin-top: 20px;">Click map to initiate scan</p>';
}

// Hook map clicks
map.on('click', function(e) {
    predictLocation(e.latlng.lat, e.latlng.lng);
});

// Boot app
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    loadAllLocations();
});