// Cosmic Hunter - Frontend JavaScript

class CosmicHunter {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentResults = [];
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

    async processFile(file) {
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showError('Please select a CSV file.');
            return;
        }

        // Check file size (limit to 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size too large. Please select a file smaller than 10MB.');
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
        } else {
            // Batch results display
            resultsContent.innerHTML = this.createBatchResultsHTML();
        }
    }

    createSingleResultHTML(result) {
        const isConfirmed = result.classification === 'CONFIRMED';
        const confidencePercent = Math.round(result.confidence * 100);
        
        return `
            <div class="result-card">
                <div class="result-header">
                    <div class="classification-badge ${isConfirmed ? 'confirmed' : 'not-confirmed'}">
                        ${result.classification}
                    </div>
                    <div class="confidence-meter">
                        <span>Confidence: ${confidencePercent}%</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                    </div>
                </div>
                
                ${result.feature_importance ? `
                    <h4>Feature Importance</h4>
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
            
            <div class="batch-results">
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
            const response = await fetch(`${this.apiBaseUrl}/api/chat`, {
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
        messageDiv.innerHTML = `<div class="message-content">${message}</div>`;
        
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
