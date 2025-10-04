// Cosmic Hunter - Frontend JavaScript

class CosmicHunter {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentResults = [];
        this.charts = {}; // Store chart instances
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupChatWidget();
        this.loadSampleData();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Manual scan form
        document.getElementById('scan-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleManualScan();
        });

        // File upload
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('csv-file');

        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadZone.addEventListener('drop', (e) => this.handleDrop(e));
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Chat functionality
        document.getElementById('chat-send').addEventListener('click', () => this.sendChatMessage());
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendChatMessage();
        });
    }

    setupChatWidget() {
        const chatToggle = document.getElementById('chat-toggle');
        const chatWindow = document.getElementById('chat-window');
        const chatClose = document.getElementById('chat-close');

        chatToggle.addEventListener('click', () => {
            chatWindow.classList.toggle('active');
        });

        chatClose.addEventListener('click', () => {
            chatWindow.classList.remove('active');
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // If switching to results tab, update the content
        if (tabName === 'results') {
            this.updateResultsDisplay();
        }
    }

    async handleManualScan() {
        const formData = new FormData(document.getElementById('scan-form'));
        const features = {};

        // Extract form data
        for (let [key, value] of formData.entries()) {
            features[key] = parseFloat(value);
        }

        // Validate form data
        if (!this.validateFormData(features)) {
            this.showError('Please fill in all required fields with valid values.');
            return;
        }

        this.showLoading(true, 'Analyzing exoplanet features...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/predict/single`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(features)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            this.currentResults = [result];
            this.switchTab('results');
            this.showSuccess('Exoplanet analysis completed successfully!');
            
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Failed to analyze exoplanet: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }
    
    resetFileInput() {
        // Reset the file input to allow re-uploading the same file
        const fileInput = document.getElementById('csv-file');
        fileInput.value = '';
    }

    async processFile(file) {
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showError('Please select a CSV file.');
            this.resetFileInput();
            return;
        }

        // Check file size (limit to 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size too large. Please select a file smaller than 10MB.');
            this.resetFileInput();
            return;
        }

        this.showLoading(true, 'Processing CSV file...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiBaseUrl}/api/predict/batch`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.results && result.results.length > 0) {
                this.currentResults = result.results;
                this.switchTab('results');
                
                const successMsg = `Successfully processed ${result.successful_predictions} out of ${result.total_processed} records`;
                this.showSuccess(successMsg);
            } else {
                throw new Error('No results returned from batch processing');
            }
            
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Failed to process CSV file: ${error.message}`);
        } finally {
            this.showLoading(false);
            this.resetFileInput();
        }
    }

    updateResultsDisplay() {
        const resultsContent = document.getElementById('results-content');
        
        if (this.currentResults.length === 0) {
            resultsContent.innerHTML = `
                <div class="no-results">
                    <div class="no-results-icon">🔭</div>
                    <h3>No Analysis Results Yet</h3>
                    <p>Run a manual scan or upload a CSV file to see results here.</p>
                </div>
            `;
            return;
        }

        if (this.currentResults.length === 1) {
            // Single result display
            const result = this.currentResults[0];
            resultsContent.innerHTML = this.createSingleResultHTML(result);
            
            // Create visualizations after HTML is rendered
            setTimeout(() => {
                this.createConfidenceGauge('confidence-gauge', result.confidence);
                if (result.feature_importance) {
                    this.createFeatureImportanceChart('feature-importance-chart', result.feature_importance);
                }
                this.addChartInteractivity();
                this.addExportButtons();
            }, 100);
        } else {
            // Batch results display
            resultsContent.innerHTML = this.createBatchResultsHTML();
            
            // Create batch visualizations after HTML is rendered
            setTimeout(() => {
                this.createClassificationPieChart('classification-pie-chart', this.currentResults);
                this.createConfidenceDistributionChart('confidence-distribution-chart', this.currentResults);
                this.addChartInteractivity();
                this.addExportButtons();
            }, 100);
        }
    }

    createSingleResultHTML(result) {
        const classification = result.classification;
        const badgeClass = 
            classification === 'CONFIRMED' ? 'confirmed' : 
            classification === 'CANDIDATE' ? 'candidate' : 
            'false-positive';
        const confidencePercent = Math.round(result.confidence * 100);
        
        return `
            <div class="result-card">
                <div class="result-header">
                    <div class="classification-badge ${badgeClass}">
                        ${result.classification}
                    </div>
                    <div class="confidence-meter">
                        <span>Confidence: ${confidencePercent}%</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Visualization Section -->
                <div class="visualization-section">
                    <div class="chart-container">
                        <div class="chart-header">
                            <h4>Confidence Gauge</h4>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="confidence-gauge"></canvas>
                        </div>
                    </div>
                    
                    ${result.feature_importance ? `
                        <div class="chart-container">
                            <div class="chart-header">
                                <h4>Feature Importance Analysis</h4>
                            </div>
                            <div class="chart-wrapper">
                                <canvas id="feature-importance-chart"></canvas>
                            </div>
                        </div>
                    ` : ''}
                </div>
                
                ${result.feature_importance ? `
                    <div class="feature-details">
                        <h4>Feature Details</h4>
                        <div class="feature-grid">
                            ${Object.entries(result.feature_importance)
                                .sort(([,a], [,b]) => b - a)
                                .map(([feature, importance]) => `
                                    <div class="feature-item">
                                        <div class="feature-label">${this.getFeatureDisplayName(feature)}</div>
                                        <div class="feature-value">${(importance * 100).toFixed(1)}%</div>
                                    </div>
                                `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                <div class="result-timestamp">
                    <small>Analysis completed: ${new Date(result.timestamp).toLocaleString()}</small>
                </div>
            </div>
        `;
    }

    createBatchResultsHTML() {
        const total = this.currentResults.length;
        const successful = this.currentResults.filter(r => !r.error).length;
        const confirmed = this.currentResults.filter(r => r.classification === 'CONFIRMED').length;
        
        return `
            <div class="batch-summary">
                <h3>Batch Analysis Summary</h3>
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-value">${total}</span>
                        <span class="stat-label">Total Processed</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${successful}</span>
                        <span class="stat-label">Successful</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${confirmed}</span>
                        <span class="stat-label">Confirmed</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${Math.round((confirmed/successful)*100)}%</span>
                        <span class="stat-label">Confirmation Rate</span>
                    </div>
                </div>
            </div>
            
            <!-- Batch Visualizations -->
            <div class="batch-visualizations">
                <div class="chart-container">
                    <div class="chart-header">
                        <h4>Classification Distribution</h4>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="classification-pie-chart"></canvas>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-header">
                        <h4>Confidence Distribution</h4>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="confidence-distribution-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="batch-results">
                <h4>Individual Results</h4>
                ${this.currentResults.slice(0, 10).map((result, index) => `
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-id">#${index + 1}</span>
                            <div class="classification-badge ${result.classification === 'CONFIRMED' ? 'confirmed' : 'not-confirmed'}">
                                ${result.classification}
                            </div>
                            <span class="confidence-value">${Math.round(result.confidence * 100)}%</span>
                        </div>
                    </div>
                `).join('')}
                
                ${total > 10 ? `<p class="more-results">... and ${total - 10} more results</p>` : ''}
            </div>
        `;
    }

    getFeatureDisplayName(feature) {
        const names = {
            'koi_ror': 'Radius Ratio',
            'koi_impact': 'Impact Parameter',
            'koi_depth': 'Transit Depth',
            'koi_prad': 'Planetary Radius',
            'koi_teq': 'Equilibrium Temp',
            'koi_duration': 'Transit Duration',
            'koi_insol': 'Insolation Flux',
            'koi_steff': 'Stellar Temperature'
        };
        return names[feature] || feature;
    }

    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;

        // Add user message to chat
        this.addChatMessage(message, 'user');
        input.value = '';

        // Show typing indicator
        const typingId = this.addTypingIndicator();

        try {
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    context: this.currentResults.length > 0 ? this.currentResults : null
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator(typingId);
            
            // Add AI response
            this.addChatMessage(result.reply, 'ai');
            
        } catch (error) {
            console.error('Error:', error);
            
            // Remove typing indicator
            this.removeTypingIndicator(typingId);
            
            // Add error message
            this.addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
        }
    }

    addChatMessage(message, sender) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        // format: keep newlines + bold
        const formattedMessage = message
            .replace(/\n/g, "<br>")
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

        messageDiv.innerHTML = `<div class="message-content">${formattedMessage}</div>`;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    addTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        const typingId = 'typing-' + Date.now();
        typingDiv.id = typingId;
        typingDiv.className = 'message ai-message typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span class="typing-text">Cosmic AI is thinking...</span>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return typingId;
    }

    removeTypingIndicator(typingId) {
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
    }

    loadSampleData() {
        // Load sample data for demonstration
        fetch(`${this.apiBaseUrl}/api/sample/data`)
            .then(response => response.json())
            .then(data => {
                // Populate form with sample data
                Object.entries(data).forEach(([key, value]) => {
                    const input = document.getElementById(key);
                    if (input) {
                        input.value = value;
                    }
                });
            })
            .catch(error => {
                console.log('Could not load sample data:', error);
            });
    }

    validateFormData(features) {
        const requiredFields = [
            'koi_ror', 'koi_impact', 'koi_depth', 'koi_prad',
            'koi_teq', 'koi_duration', 'koi_insol', 'koi_steff'
        ];
        
        for (const field of requiredFields) {
            if (!features[field] || isNaN(features[field])) {
                return false;
            }
        }
        
        // Additional validation ranges
        if (features.koi_ror < 0.001 || features.koi_ror > 1.0) return false;
        if (features.koi_impact < 0.0 || features.koi_impact > 1.0) return false;
        if (features.koi_depth < 0 || features.koi_depth > 1000) return false;
        if (features.koi_prad < 0.1 || features.koi_prad > 50.0) return false;
        if (features.koi_teq < 100 || features.koi_teq > 5000) return false;
        if (features.koi_duration < 0.1 || features.koi_duration > 50.0) return false;
        if (features.koi_insol < 0.1 || features.koi_insol > 10000) return false;
        if (features.koi_steff < 2000 || features.koi_steff > 10000) return false;
        
        return true;
    }

    showLoading(show, message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        const loadingText = overlay.querySelector('h3');
        const loadingSubtext = overlay.querySelector('p');
        
        if (show) {
            loadingText.textContent = message;
            loadingSubtext.textContent = 'Please wait while we process your request';
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showNotification(message, type = 'info') {
        const notificationDiv = document.createElement('div');
        notificationDiv.className = `notification ${type}-notification`;
        
        const colors = {
            error: 'rgba(239, 68, 68, 0.9)',
            success: 'rgba(16, 185, 129, 0.9)',
            info: 'rgba(0, 212, 255, 0.9)'
        };
        
        const icons = {
            error: '❌',
            success: '✅',
            info: 'ℹ️'
        };
        
        notificationDiv.style.cssText = `
            position: fixed;
            top: 2rem;
            right: 2rem;
            background: ${colors[type]};
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            z-index: 3000;
            font-family: 'Orbitron', monospace;
            font-weight: 700;
            box-shadow: 0 4px 20px ${colors[type].replace('0.9', '0.3')};
            display: flex;
            align-items: center;
            gap: 0.5rem;
            max-width: 400px;
            animation: slideIn 0.3s ease-out;
        `;
        
        notificationDiv.innerHTML = `
            <span class="notification-icon">${icons[type]}</span>
            <span class="notification-text">${message}</span>
        `;
        
        document.body.appendChild(notificationDiv);
        
        setTimeout(() => {
            notificationDiv.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                notificationDiv.remove();
            }, 300);
        }, 5000);
    }

    // Visualization Functions
    destroyChart(chartId) {
        if (this.charts[chartId]) {
            this.charts[chartId].destroy();
            delete this.charts[chartId];
        }
    }

    createFeatureImportanceChart(containerId, featureData) {
        this.destroyChart(containerId);
        
        const ctx = document.getElementById(containerId).getContext('2d');
        const sortedFeatures = Object.entries(featureData)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8); // Top 8 features

        this.charts[containerId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sortedFeatures.map(([feature]) => this.getFeatureDisplayName(feature)),
                datasets: [{
                    label: 'Importance (%)',
                    data: sortedFeatures.map(([, importance]) => importance * 100),
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 2,
                    borderRadius: 4,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Feature Importance Analysis',
                        color: '#00d4ff',
                        font: {
                            family: 'Orbitron',
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 46, 0.9)',
                        titleColor: '#00d4ff',
                        bodyColor: '#b0b0b0',
                        borderColor: '#00d4ff',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                        titleFont: {
                            family: 'Orbitron',
                            weight: 'bold'
                        },
                        bodyFont: {
                            family: 'Orbitron'
                        },
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return `Importance: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            color: 'rgba(26, 26, 46, 0.3)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron',
                                size: 10
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                },
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        const elementIndex = elements[0].index;
                        const featureName = sortedFeatures[elementIndex][0];
                        this.showFeatureDetails(featureName, sortedFeatures[elementIndex][1]);
                    }
                }
            }
        });
    }

    createConfidenceGauge(containerId, confidence) {
        this.destroyChart(containerId);
        
        const ctx = document.getElementById(containerId).getContext('2d');
        const percentage = Math.round(confidence * 100);
        
        this.charts[containerId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [percentage, 100 - percentage],
                    backgroundColor: [
                        'rgba(0, 212, 255, 0.8)',
                        'rgba(26, 26, 46, 0.3)'
                    ],
                    borderColor: [
                        'rgba(0, 212, 255, 1)',
                        'rgba(26, 26, 46, 0.5)'
                    ],
                    borderWidth: 2,
                    cutout: '70%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: true,
                        ...this.getCosmicTooltipConfig(),
                        callbacks: {
                            label: function(context) {
                                return `Confidence: ${context.parsed}%`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });

        // Add center text
        const centerText = document.createElement('div');
        centerText.className = 'gauge-center-text';
        centerText.innerHTML = `
            <div class="gauge-percentage">${percentage}%</div>
            <div class="gauge-label">Confidence</div>
        `;
        centerText.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #00d4ff;
            font-family: 'Orbitron', monospace;
            font-weight: bold;
        `;
        
        const gaugeContainer = document.getElementById(containerId).parentElement;
        gaugeContainer.style.position = 'relative';
        gaugeContainer.appendChild(centerText);
    }

    createClassificationPieChart(containerId, results) {
        this.destroyChart(containerId);
        
        const ctx = document.getElementById(containerId).getContext('2d');
        const confirmed = results.filter(r => r.classification === 'CONFIRMED').length;
        const candidate = results.filter(r => r.classification === 'CANDIDATE').length;
        const falsePositive = results.filter(r => r.classification === 'FALSE POSITIVE').length;
        
        this.charts[containerId] = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Confirmed', 'Candidate', 'False Positive'],
                datasets: [{
                    data: [confirmed, candidate, falsePositive],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(239, 68, 68, 0.8)'
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(239, 68, 68, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron',
                                size: 12
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Classification Distribution',
                        color: '#00d4ff',
                        font: {
                            family: 'Orbitron',
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        ...this.getCosmicTooltipConfig(),
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    createConfidenceDistributionChart(containerId, results) {
        this.destroyChart(containerId);
        
        const ctx = document.getElementById(containerId).getContext('2d');
        const confidences = results.map(r => Math.round(r.confidence * 100));
        
        // Create histogram bins
        const bins = [0, 20, 40, 60, 80, 100];
        const binLabels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'];
        const binCounts = new Array(bins.length - 1).fill(0);
        
        confidences.forEach(conf => {
            for (let i = 0; i < bins.length - 1; i++) {
                if (conf >= bins[i] && conf < bins[i + 1]) {
                    binCounts[i]++;
                    break;
                }
            }
        });
        
        // Color bins based on our classification thresholds
        const backgroundColors = [
            'rgba(239, 68, 68, 0.6)',  // 0-20% (False Positive)
            'rgba(239, 68, 68, 0.6)',  // 20-40% (False Positive)
            'rgba(245, 158, 11, 0.6)', // 40-60% (Candidate)
            'rgba(245, 158, 11, 0.6)', // 60-80% (Candidate)
            'rgba(16, 185, 129, 0.6)'  // 80-100% (Confirmed)
        ];
        
        const borderColors = [
            'rgba(239, 68, 68, 1)',  // 0-20% (False Positive)
            'rgba(239, 68, 68, 1)',  // 20-40% (False Positive)
            'rgba(245, 158, 11, 1)', // 40-60% (Candidate)
            'rgba(245, 158, 11, 1)', // 60-80% (Candidate)
            'rgba(16, 185, 129, 1)'  // 80-100% (Confirmed)
        ];
        
        this.charts[containerId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: 'Number of Results',
                    data: binCounts,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 4,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Confidence Distribution',
                        color: '#00d4ff',
                        font: {
                            family: 'Orbitron',
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            color: 'rgba(26, 26, 46, 0.3)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    createRadarChart(containerId, features) {
        this.destroyChart(containerId);
        
        const ctx = document.getElementById(containerId).getContext('2d');
        const featureNames = Object.keys(features);
        const featureValues = Object.values(features);
        
        // Normalize values to 0-100 scale for radar chart
        const normalizedValues = featureValues.map(value => {
            // Simple normalization - adjust based on typical ranges
            return Math.min(100, Math.max(0, (value / Math.max(...featureValues)) * 100));
        });
        
        this.charts[containerId] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: featureNames.map(name => this.getFeatureDisplayName(name)),
                datasets: [{
                    label: 'Feature Values',
                    data: normalizedValues,
                    backgroundColor: 'rgba(0, 212, 255, 0.2)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(0, 212, 255, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(0, 212, 255, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Feature Profile',
                        color: '#00d4ff',
                        font: {
                            family: 'Orbitron',
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            color: 'rgba(26, 26, 46, 0.3)'
                        },
                        pointLabels: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron',
                                size: 10
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    createScatterPlot(containerId, results, xFeature, yFeature) {
        this.destroyChart(containerId);
        
        const ctx = document.getElementById(containerId).getContext('2d');
        const confirmedData = [];
        const notConfirmedData = [];
        
        results.forEach(result => {
            const point = {
                x: result[xFeature] || 0,
                y: result[yFeature] || 0
            };
            
            if (result.classification === 'CONFIRMED') {
                confirmedData.push(point);
            } else {
                notConfirmedData.push(point);
            }
        });
        
        this.charts[containerId] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Confirmed',
                    data: confirmedData,
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 2
                }, {
                    label: 'Not Confirmed',
                    data: notConfirmedData,
                    backgroundColor: 'rgba(239, 68, 68, 0.6)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: `${this.getFeatureDisplayName(xFeature)} vs ${this.getFeatureDisplayName(yFeature)}`,
                        color: '#00d4ff',
                        font: {
                            family: 'Orbitron',
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: this.getFeatureDisplayName(xFeature),
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        ticks: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            color: 'rgba(26, 26, 46, 0.3)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: this.getFeatureDisplayName(yFeature),
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        ticks: {
                            color: '#b0b0b0',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            color: 'rgba(26, 26, 46, 0.3)'
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    // Interactive Features
    showFeatureDetails(featureName, importance) {
        const featureDescriptions = {
            'koi_ror': 'Planet-to-Star Radius Ratio: The ratio of the planet\'s radius to the host star\'s radius. This is crucial for determining the planet\'s size relative to its star.',
            'koi_impact': 'Impact Parameter: The minimum distance between the center of the planet and the center of the star during transit, normalized by the star\'s radius.',
            'koi_depth': 'Transit Depth: The fractional decrease in stellar brightness during transit, measured in parts per million (ppm).',
            'koi_prad': 'Planetary Radius: The radius of the planet measured in Earth radii (R⊕). This helps classify planets by size.',
            'koi_teq': 'Equilibrium Temperature: The theoretical temperature of the planet assuming it\'s a black body in thermal equilibrium with its star.',
            'koi_duration': 'Transit Duration: The time it takes for the planet to completely cross the star\'s disk during transit.',
            'koi_insol': 'Insolation Flux: The amount of stellar radiation received by the planet per unit area, relative to Earth\'s insolation.',
            'koi_steff': 'Stellar Effective Temperature: The surface temperature of the host star, which affects the planet\'s climate and habitability.'
        };

        const description = featureDescriptions[featureName] || 'Feature description not available.';
        const displayName = this.getFeatureDisplayName(featureName);
        const importancePercent = (importance * 100).toFixed(1);

        this.showNotification(
            `<strong>${displayName}</strong><br>Importance: ${importancePercent}%<br><br>${description}`,
            'info'
        );
    }

    addChartInteractivity() {
        // Add click handlers to all charts for enhanced interactivity
        Object.keys(this.charts).forEach(chartId => {
            const chart = this.charts[chartId];
            if (chart && chart.canvas) {
                chart.canvas.style.cursor = 'pointer';
            }
        });
    }

    // Enhanced tooltip configuration for all charts
    getCosmicTooltipConfig() {
        return {
            backgroundColor: 'rgba(26, 26, 46, 0.95)',
            titleColor: '#00d4ff',
            bodyColor: '#b0b0b0',
            borderColor: '#00d4ff',
            borderWidth: 1,
            cornerRadius: 8,
            titleFont: {
                family: 'Orbitron',
                weight: 'bold',
                size: 12
            },
            bodyFont: {
                family: 'Orbitron',
                size: 11
            },
            padding: 12,
            displayColors: false
        };
    }

    // Export functionality for charts
    exportChart(chartId, filename = 'chart') {
        const chart = this.charts[chartId];
        if (chart) {
            const url = chart.toBase64Image();
            const link = document.createElement('a');
            link.download = `${filename}.png`;
            link.href = url;
            link.click();
            this.showSuccess('Chart exported successfully!');
        }
    }

    // Add export buttons to charts
    addExportButtons() {
        Object.keys(this.charts).forEach(chartId => {
            const chartContainer = document.getElementById(chartId).parentElement;
            if (!chartContainer.querySelector('.export-button')) {
                const exportBtn = document.createElement('button');
                exportBtn.className = 'export-button';
                exportBtn.innerHTML = '📊 Export';
                exportBtn.style.cssText = `
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: rgba(0, 212, 255, 0.2);
                    border: 1px solid rgba(0, 212, 255, 0.5);
                    color: #00d4ff;
                    padding: 0.5rem 1rem;
                    border-radius: 6px;
                    font-family: 'Orbitron', monospace;
                    font-size: 0.8rem;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    z-index: 10;
                `;
                
                exportBtn.addEventListener('click', () => {
                    this.exportChart(chartId, `exoplanet-${chartId}`);
                });
                
                exportBtn.addEventListener('mouseenter', () => {
                    exportBtn.style.background = 'rgba(0, 212, 255, 0.3)';
                    exportBtn.style.transform = 'scale(1.05)';
                });
                
                exportBtn.addEventListener('mouseleave', () => {
                    exportBtn.style.background = 'rgba(0, 212, 255, 0.2)';
                    exportBtn.style.transform = 'scale(1)';
                });
                
                chartContainer.style.position = 'relative';
                chartContainer.appendChild(exportBtn);
            }
        });
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CosmicHunter();
});

// Add some additional CSS for batch results
const additionalStyles = `
    .batch-summary {
        background: rgba(10, 10, 15, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
    }
    
    .summary-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: rgba(26, 26, 46, 0.6);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .stat-value {
        display: block;
        font-size: 2rem;
        font-weight: 900;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .batch-results {
        max-height: 400px;
        overflow-y: auto;
    }
    
    .result-id {
        color: var(--text-secondary);
        font-weight: 700;
    }
    
    .confidence-value {
        color: var(--primary-color);
        font-weight: 700;
    }
    
    .more-results {
        text-align: center;
        color: var(--text-secondary);
        font-style: italic;
        margin-top: 1rem;
    }
    
    .result-timestamp {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
    }
    
    .result-timestamp small {
        color: var(--text-muted);
    }
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);
