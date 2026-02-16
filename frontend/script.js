// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Asset database for search
const ASSET_DATABASE = [
    // US Stocks
    { symbol: 'SPY', name: 'S&P 500 ETF' },
    { symbol: 'QQQ', name: 'NASDAQ 100 ETF' },
    { symbol: 'IWM', name: 'Russell 2000 ETF' },
    { symbol: 'DIA', name: 'Dow Jones ETF' },
    { symbol: 'VTI', name: 'Total Stock Market ETF' },
    { symbol: 'VOO', name: 'S&P 500 ETF (Vanguard)' },
    { symbol: 'IVV', name: 'S&P 500 ETF (iShares)' },

    // Tech Stocks
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corp.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.' },
    { symbol: 'META', name: 'Meta Platforms Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'NFLX', name: 'Netflix Inc.' },

    // Bonds
    { symbol: 'TLT', name: '20+ Year Treasury Bond ETF' },
    { symbol: 'IEF', name: '7-10 Year Treasury Bond ETF' },
    { symbol: 'SHY', name: '1-3 Year Treasury Bond ETF' },
    { symbol: 'BND', name: 'Total Bond Market ETF' },
    { symbol: 'LQD', name: 'Corporate Bond ETF' },
    { symbol: 'HYG', name: 'High Yield Bond ETF' },

    // Commodities
    { symbol: 'GLD', name: 'Gold Trust ETF' },
    { symbol: 'SLV', name: 'Silver Trust ETF' },
    { symbol: 'USO', name: 'United States Oil Fund' },
    { symbol: 'UNG', name: 'United States Natural Gas Fund' },
    { symbol: 'WEAT', name: 'Wheat ETF' },
    { symbol: 'CORN', name: 'Corn ETF' },

    // Real Estate
    { symbol: 'VNQ', name: 'Real Estate ETF' },
    { symbol: 'REET', name: 'Global Real Estate ETF' },
    { symbol: 'IYR', name: 'US Real Estate ETF' },

    // International
    { symbol: 'EFA', name: 'EAFE International ETF' },
    { symbol: 'EEM', name: 'Emerging Markets ETF' },
    { symbol: 'VXUS', name: 'Total International Stock ETF' },
    { symbol: 'FXI', name: 'China Large-Cap ETF' },
    { symbol: 'EWJ', name: 'Japan ETF' },

    // Crypto
    { symbol: 'BTC-USD', name: 'Bitcoin USD' },
    { symbol: 'ETH-USD', name: 'Ethereum USD' },
    { symbol: 'BNB-USD', name: 'Binance Coin USD' },
    { symbol: 'SOL-USD', name: 'Solana USD' },
    { symbol: 'ADA-USD', name: 'Cardano USD' },

    // Sector ETFs
    { symbol: 'XLF', name: 'Financial Sector ETF' },
    { symbol: 'XLK', name: 'Technology Sector ETF' },
    { symbol: 'XLV', name: 'Healthcare Sector ETF' },
    { symbol: 'XLE', name: 'Energy Sector ETF' },
    { symbol: 'XLI', name: 'Industrial Sector ETF' },
    { symbol: 'XLP', name: 'Consumer Staples ETF' },
    { symbol: 'XLY', name: 'Consumer Discretionary ETF' },
    { symbol: 'XLU', name: 'Utilities Sector ETF' },
    { symbol: 'XLRE', name: 'Real Estate Sector ETF' },
];

// State management
let portfolioChart = null;
let drawdownChart = null;
let allocationChart = null;
let volatilityChart = null;
let volumeChart = null;
let backtestComparisonChart = null;
let stressChart = null;
let searchTimeout = null;

// Initialize event listeners
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('portfolio-form');
    const runBtn = document.getElementById('run-portfolio');
    const backtestBtn = document.getElementById('run-backtest');
    const stressTestBtn = document.getElementById('run-stress-test');
    const searchInput = document.getElementById('asset-search');

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        runPortfolio();
    });

    runBtn.addEventListener('click', runPortfolio);
    backtestBtn.addEventListener('click', runBacktest);
    stressTestBtn.addEventListener('click', runStressTest);

    // Asset search functionality
    searchInput.addEventListener('input', handleAssetSearch);
    searchInput.addEventListener('focus', handleAssetSearch);

    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.asset-selector-container')) {
            hideSearchResults();
        }
    });

    // Set default dates
    setDefaultDates();

    // Check API health
    checkApiHealth();
});

// Asset management functions
function handleAssetSearch() {
    const searchTerm = document.getElementById('asset-search').value.toLowerCase().trim();

    // Clear previous timeout
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }

    // Set new timeout for debouncing
    searchTimeout = setTimeout(() => {
        if (searchTerm.length < 1) {
            hideSearchResults();
            return;
        }

        // Filter assets
        const results = ASSET_DATABASE.filter(asset =>
            asset.symbol.toLowerCase().includes(searchTerm) ||
            asset.name.toLowerCase().includes(searchTerm)
        ).slice(0, 10); // Limit to 10 results

        displaySearchResults(results);
    }, 300);
}

function displaySearchResults(results) {
    const resultsDiv = document.getElementById('asset-search-results');

    if (results.length === 0) {
        resultsDiv.innerHTML = '<div class="search-result-item">No assets found</div>';
    } else {
        resultsDiv.innerHTML = results.map(asset => `
            <div class="search-result-item" onclick="addAsset('${asset.symbol}')">
                <span class="symbol">${asset.symbol}</span>
                <span class="name">${asset.name}</span>
            </div>
        `).join('');
    }

    resultsDiv.classList.remove('hidden');
}

function hideSearchResults() {
    const resultsDiv = document.getElementById('asset-search-results');
    resultsDiv.classList.add('hidden');
}

function addAsset(symbol) {
    // Check if asset already exists
    const existingAssets = document.querySelectorAll('.asset-tag');
    for (let asset of existingAssets) {
        if (asset.dataset.symbol === symbol) {
            alert(`${symbol} is already selected`);
            document.getElementById('asset-search').value = '';
            hideSearchResults();
            return;
        }
    }

    // Add new asset tag
    const selectedAssets = document.getElementById('selected-assets');
    const assetTag = document.createElement('div');
    assetTag.className = 'asset-tag';
    assetTag.dataset.symbol = symbol;
    assetTag.innerHTML = `
        ${symbol}
        <span class="remove-asset" onclick="removeAsset('${symbol}')">×</span>
    `;
    selectedAssets.appendChild(assetTag);

    // Clear search
    document.getElementById('asset-search').value = '';
    hideSearchResults();
}

function removeAsset(symbol) {
    const assetTag = document.querySelector(`.asset-tag[data-symbol="${symbol}"]`);
    if (assetTag) {
        assetTag.remove();
    }
}

function getSelectedAssets() {
    const assets = [];
    document.querySelectorAll('.asset-tag').forEach(tag => {
        assets.push(tag.dataset.symbol);
    });
    return assets;
}

async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('API is healthy');
        } else {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.error('API not reachable. Make sure the backend server is running on port 8000');
        showError('Cannot connect to backend server. Please ensure it is running on port 8000.');
    }
}

function setDefaultDates() {
    const today = new Date();
    const endDate = today.toISOString().split('T')[0];
    const startDate = new Date(today.setFullYear(today.getFullYear() - 1)).toISOString().split('T')[0];

    document.getElementById('start-date').value = startDate;
    document.getElementById('end-date').value = endDate;

    console.log(`Default dates set: ${startDate} to ${endDate}`);
}

function validateDates() {
    const startDate = new Date(document.getElementById('start-date').value);
    const endDate = new Date(document.getElementById('end-date').value);
    const today = new Date();

    today.setHours(0, 0, 0, 0);
    startDate.setHours(0, 0, 0, 0);
    endDate.setHours(0, 0, 0, 0);

    if (endDate > today) {
        alert('End date cannot be in the future. Please select a date up to today.');
        document.getElementById('end-date').value = today.toISOString().split('T')[0];
        return false;
    }

    if (startDate >= endDate) {
        alert('Start date must be before end date');
        return false;
    }

    const dayDiff = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));
    if (dayDiff < 30) {
        alert('Date range should be at least 30 days for meaningful analysis');
        return false;
    }

    return true;
}

function collectFormData() {
    const assets = getSelectedAssets();

    if (assets.length === 0) {
        alert('Please select at least one asset');
        throw new Error('No assets selected');
    }

    return {
        assets: assets,
        start_date: document.getElementById('start-date').value,
        end_date: document.getElementById('end-date').value,
        initial_capital: parseFloat(document.getElementById('initial-capital').value),
        risk_profile: document.getElementById('risk-profile').value,
        rebalance_frequency: document.getElementById('rebalance-frequency').value,
        enable_risk_management: document.getElementById('enable-risk').checked
    };
}

async function runPortfolio() {
    if (!validateDates()) {
        return;
    }

    showLoading();
    hideResults();

    try {
        const requestData = collectFormData();
        console.log('Sending portfolio request:', requestData);

        const response = await fetch(`${API_BASE_URL}/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        const responseData = await response.json();
        console.log('Portfolio response:', responseData);

        if (!response.ok) {
            throw new Error(responseData.detail || `HTTP error! status: ${response.status}`);
        }

        if (responseData.success) {
            displayResults(responseData);
        } else {
            throw new Error('Failed to run portfolio');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to backend.');
    } finally {
        hideLoading();
    }
}

async function runBacktest() {
    if (!validateDates()) {
        return;
    }

    showLoading();
    hideResults();

    try {
        const requestData = collectFormData();
        console.log('Sending backtest request:', requestData);

        const response = await fetch(`${API_BASE_URL}/backtest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        const responseData = await response.json();
        console.log('Backtest response:', responseData);

        if (!response.ok) {
            throw new Error(responseData.detail || `HTTP error! status: ${response.status}`);
        }

        if (responseData.success) {
            displayBacktestResults(responseData);
        } else {
            throw new Error('Failed to run backtest');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to backend.');
    } finally {
        hideLoading();
    }
}

async function runStressTest() {
    if (!validateDates()) {
        return;
    }

    showLoading();
    hideResults();

    try {
        // Show shock type selection in a more user-friendly way
        const shockType = prompt('Enter shock type:\n- crash\n- volatility_spike\n- correlation_spike', 'crash');
        if (!shockType) {
            hideLoading();
            return;
        }

        const shockMagnitude = parseFloat(prompt('Enter shock magnitude (0-1):', '0.3'));
        if (isNaN(shockMagnitude)) {
            hideLoading();
            return;
        }

        const requestData = {
            ...collectFormData(),
            shock_type: shockType,
            shock_magnitude: shockMagnitude
        };

        console.log('Sending stress test request:', requestData);

        const response = await fetch(`${API_BASE_URL}/stress_test`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        const responseData = await response.json();
        console.log('Stress test response:', responseData);

        if (!response.ok) {
            throw new Error(responseData.detail || `HTTP error! status: ${response.status}`);
        }

        if (responseData.success) {
            displayStressTestResults(responseData);
        } else {
            throw new Error('Failed to run stress test');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to backend.');
    } finally {
        hideLoading();
    }
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    disableButtons(true);
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
    disableButtons(false);
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
    // Remove any backtest comparison tables
    const backtestComparison = document.querySelector('.backtest-comparison');
    if (backtestComparison) {
        backtestComparison.remove();
    }
    // Remove any stress test results
    const stressResults = document.querySelector('.stress-test-results');
    if (stressResults) {
        stressResults.remove();
    }
}

function disableButtons(disabled) {
    document.querySelectorAll('button').forEach(btn => {
        btn.disabled = disabled;
    });
}

function showError(message) {
    alert('Error: ' + message);
}

function displayResults(data) {
    document.getElementById('results').classList.remove('hidden');

    // Update metrics
    document.getElementById('final-value').textContent = formatCurrency(data.metrics.final_value);
    document.getElementById('total-return').textContent = formatPercentage(data.metrics.total_return);
    document.getElementById('sharpe-ratio').textContent = data.metrics.sharpe_ratio.toFixed(2);
    document.getElementById('max-drawdown').textContent = formatPercentage(data.metrics.max_drawdown);

    // Create charts
    // Update Regime Badge
    if (data.regimes && data.regimes.length > 0) {
        const lastRegime = data.regimes[data.regimes.length - 1];
        const regimeBadge = document.getElementById('current-regime');
        regimeBadge.textContent = lastRegime.regime;
        // Handle class for styling (replace spaces with hyphens)
        // Remove old regime classes
        regimeBadge.className = 'metric-value regime-badge';
        const regimeClass = lastRegime.regime.replace(/\s+/g, '-');
        regimeBadge.classList.add(regimeClass);
    }

    // Create charts
    createPortfolioChart(data.portfolio_values);
    createDrawdownChart(data.drawdowns);
    createAllocationChart(data.current_allocation);
    createVolatilityChart(data.regimes);

    // Create Volume Chart
    if (data.volumes && Object.keys(data.volumes).length > 0) {
        // Use dates from regimes as labels if available, assuming alignment
        const dates = data.regimes.map(r => r.date);
        createVolumeChart(data.volumes, dates);
    } else {
        if (volumeChart) {
            volumeChart.destroy();
            volumeChart = null;
        }
    }

    // Display explanations
    displayExplanations(data.explanations);

    // Display AI Analysis
    const aiContainer = document.getElementById('ai-analysis');
    if (data.ai_analysis) {
        // Convert newlines to breaks and styling markdown-like bold
        let formattedText = data.ai_analysis
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');

        aiContainer.innerHTML = `<p class="params-text">${formattedText}</p>`;

        // Add subtle typewriter effect class (optional, purely CSS)
        aiContainer.classList.add('fade-in');
    } else {
        aiContainer.innerHTML = '<p class="placeholder-text">Analysis unavailable.</p>';
    }

    // Display regime timeline
    displayRegimeTimeline(data.regimes);
}

function displayBacktestResults(data) {
    document.getElementById('results').classList.remove('hidden');

    // Remove any existing comparison table
    const existingComparison = document.querySelector('.backtest-comparison');
    if (existingComparison) {
        existingComparison.remove();
    }

    // Log the data structure for debugging
    console.log('Backtest data structure:', data);

    // Check if we have any metrics
    const hasWithRisk = data.metrics_with_risk && Object.keys(data.metrics_with_risk).length > 0;
    const hasWithoutRisk = data.metrics_without_risk && Object.keys(data.metrics_without_risk).length > 0;

    if (!hasWithRisk && !hasWithoutRisk) {
        // Show a message that no metrics are available
        const messageHTML = `
            <div class="card backtest-comparison">
                <h3>Backtest Results</h3>
                <p>Not enough data for rolling backtest with current parameters.</p>
                <p>Suggestions:</p>
                <ul style="margin-top: 10px; margin-left: 20px;">
                    <li>Use a longer date range (at least 2 years)</li>
                    <li>Reduce the window size in the request</li>
                    <li>Select different assets</li>
                </ul>
                <p style="margin-top: 15px; color: #666;">Current data points: 261 days (sample data)</p>
            </div>
        `;

        const summaryCards = document.querySelector('.summary-cards');
        summaryCards.insertAdjacentHTML('afterend', messageHTML);
        return;
    }

    // Create a table with the backtest results
    let html = `
        <div class="card backtest-comparison">
            <h3>Backtest Results: With vs Without Risk Management</h3>
            <table style="width:100%; border-collapse: collapse; margin-top: 20px;">
                <thead>
                    <tr style="background: #f0f0f0;">
                        <th style="padding: 12px; text-align: left;">Metric</th>
                        <th style="padding: 12px; text-align: right;">With Risk Mgmt</th>
                        <th style="padding: 12px; text-align: right;">Without Risk Mgmt</th>
                        <th style="padding: 12px; text-align: right;">Improvement</th>
                    </tr>
                </thead>
                <tbody>
    `;

    // Define metrics to display
    const metrics = [
        { key: 'total_return', label: 'Total Return', format: 'percentage' },
        { key: 'cagr', label: 'CAGR', format: 'percentage' },
        { key: 'sharpe_ratio', label: 'Sharpe Ratio', format: 'number' },
        { key: 'sortino_ratio', label: 'Sortino Ratio', format: 'number' },
        { key: 'volatility', label: 'Volatility', format: 'percentage' },
        { key: 'max_drawdown', label: 'Max Drawdown', format: 'percentage' },
        { key: 'calmar_ratio', label: 'Calmar Ratio', format: 'number' },
        { key: 'win_rate', label: 'Win Rate', format: 'percentage' }
    ];

    metrics.forEach(metric => {
        const withVal = data.metrics_with_risk?.[metric.key];
        const withoutVal = data.metrics_without_risk?.[metric.key];

        if (withVal !== undefined || withoutVal !== undefined) {
            const withFormatted = formatMetricAdvanced(withVal, metric.format);
            const withoutFormatted = formatMetricAdvanced(withoutVal, metric.format);

            // Calculate improvement if both values exist
            let improvement = 'N/A';
            let improvementColor = '#666';

            if (withVal !== undefined && withoutVal !== undefined && withoutVal !== 0) {
                let impValue;
                if (metric.key === 'max_drawdown') {
                    // For drawdown, less negative is better
                    impValue = (withoutVal - withVal) / Math.abs(withoutVal);
                } else {
                    impValue = (withVal - withoutVal) / Math.abs(withoutVal);
                }

                if (!isNaN(impValue) && isFinite(impValue)) {
                    improvement = formatPercentage(impValue);
                    improvementColor = impValue > 0 ? '#27ae60' : '#e74c3c';
                }
            }

            html += `
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 10px;">${metric.label}</td>
                    <td style="padding: 10px; text-align: right;">${withFormatted}</td>
                    <td style="padding: 10px; text-align: right;">${withoutFormatted}</td>
                    <td style="padding: 10px; text-align: right; color: ${improvementColor};">${improvement}</td>
                </tr>
            `;
        }
    });

    // Add summary if available
    if (data.comparison && data.comparison.summary) {
        html += `
            <tr>
                <td colspan="4" style="padding: 15px; background: #f8f9fa; border-radius: 5px;">
                    <strong>Summary:</strong> ${data.comparison.summary.recommendation || 'Analysis complete'}
                    ${data.comparison.summary.avg_improvement_pct ? ` (Average improvement: ${data.comparison.summary.avg_improvement_pct})` : ''}
                </td>
            </tr>
        `;
    }

    html += `
                </tbody>
            </table>
            
            <div class="chart-container full-width" style="margin-top: 30px; height: 400px;">
                <h3>Rolling Window Returns Comparison</h3>
                <canvas id="backtest-comparison-chart"></canvas>
            </div>
        </div>
    `;

    // Insert comparison after summary cards
    const summaryCards = document.querySelector('.summary-cards');
    summaryCards.insertAdjacentHTML('afterend', html);

    // Create comparison chart
    if (data.rolling_values) {
        createBacktestComparisonChart(data.rolling_values);
    }
}

function displayStressTestResults(data) {
    document.getElementById('results').classList.remove('hidden');

    // Remove any existing stress test results
    const existingStress = document.querySelector('.stress-test-results');
    if (existingStress) {
        existingStress.remove();
    }

    // Create stress test results card
    const shockTypeColors = {
        'crash': '#e74c3c',
        'volatility_spike': '#f39c12',
        'correlation_spike': '#9b59b6'
    };

    const shockColor = shockTypeColors[data.shock_type] || '#667eea';

    const html = `
        <div class="card stress-test-results" style="margin-top: 20px; border-left: 4px solid ${shockColor};">
            <h3 style="color: ${shockColor};">Stress Test Results: ${data.shock_type.replace('_', ' ').toUpperCase()}</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Final Value</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">${formatCurrency(data.final_value)}</div>
                </div>
                
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Preservation Ratio</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: ${data.preservation_ratio > 0.7 ? '#27ae60' : '#e74c3c'};">${formatPercentage(data.preservation_ratio)}</div>
                </div>
                
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Max Drawdown</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #e74c3c;">${formatPercentage(data.max_drawdown)}</div>
                </div>
                
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Recovery Possible</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: ${data.recovery_possible ? '#27ae60' : '#e74c3c'};">${data.recovery_possible ? 'Yes' : 'No'}</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #f0f7ff; border-radius: 8px;">
                <strong>Scenario:</strong> ${data.explanation || `${data.shock_type} with magnitude ${data.shock_magnitude}`}
            </div>
            
            <div style="margin-top: 15px; display: flex; gap: 10px; justify-content: flex-end;">
                <button onclick="createStressChart(${JSON.stringify(data.portfolio_values)})" class="btn-secondary" style="padding: 8px 15px;">View Chart</button>
                <button onclick="hideStressResults()" class="btn-secondary" style="padding: 8px 15px;">Close</button>
            </div>
            
            <div id="stress-chart-container" style="margin-top: 20px; height: 300px; display: none;">
                <canvas id="stress-chart"></canvas>
            </div>
        </div>
    `;

    // Insert after summary cards
    const summaryCards = document.querySelector('.summary-cards');
    summaryCards.insertAdjacentHTML('afterend', html);
}

// Function to create stress test chart
function createStressChart(portfolioValues) {
    const container = document.getElementById('stress-chart-container');
    container.style.display = 'block';

    const ctx = document.getElementById('stress-chart').getContext('2d');

    if (stressChart) {
        stressChart.destroy();
    }

    stressChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: portfolioValues.length }, (_, i) => i),
            datasets: [{
                label: 'Portfolio Value Under Stress',
                data: portfolioValues,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Value: ${formatCurrency(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function (value) {
                            return formatCurrency(value);
                        }
                    }
                }
            }
        }
    });
}

// Function to hide stress results
function hideStressResults() {
    const stressResults = document.querySelector('.stress-test-results');
    if (stressResults) {
        stressResults.remove();
    }
}

// Helper function for advanced metric formatting
function formatMetricAdvanced(value, formatType) {
    if (value === undefined || value === null) return 'N/A';

    if (formatType === 'percentage') {
        return formatPercentage(value);
    } else if (formatType === 'number') {
        return value.toFixed(2);
    } else {
        return value.toString();
    }
}

function formatMetric(value, metric) {
    if (value === undefined || value === null) {
        return 'N/A';
    }
    if (metric.includes('return') || metric.includes('drawdown') || metric.includes('ratio')) {
        return formatPercentage(value);
    }
    return typeof value === 'number' ? value.toFixed(3) : value;
}

function createPortfolioChart(values) {
    const ctx = document.getElementById('portfolio-chart').getContext('2d');

    if (portfolioChart) {
        portfolioChart.destroy();
    }

    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: values.length }, (_, i) => i),
            datasets: [{
                label: 'Portfolio Value',
                data: values,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Value: ${formatCurrency(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function (value) {
                            return formatCurrency(value);
                        }
                    }
                }
            }
        }
    });
}

function createDrawdownChart(drawdowns) {
    const ctx = document.getElementById('drawdown-chart').getContext('2d');

    if (drawdownChart) {
        drawdownChart.destroy();
    }

    drawdownChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: drawdowns.length }, (_, i) => i),
            datasets: [{
                label: 'Drawdown',
                data: drawdowns,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Drawdown: ${formatPercentage(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function (value) {
                            return formatPercentage(value);
                        }
                    }
                }
            }
        }
    });
}

function createAllocationChart(allocation) {
    const ctx = document.getElementById('allocation-chart').getContext('2d');

    if (allocationChart) {
        allocationChart.destroy();
    }

    const colors = [
        '#667eea', '#764ba2', '#e74c3c', '#f39c12',
        '#27ae60', '#3498db', '#9b59b6', '#e67e22'
    ];

    const labels = Object.keys(allocation);
    const data = Object.values(allocation);

    allocationChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors.slice(0, labels.length),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ${formatPercentage(context.raw)}`;
                        }
                    }
                }
            }
        }
    });
}

function createVolatilityChart(regimes) {
    const ctx = document.getElementById('volatility-chart').getContext('2d');

    if (volatilityChart) {
        volatilityChart.destroy();
    }

    const volatilities = regimes.map(r => r.volatility || 0);
    const labels = regimes.map(r => r.date || '');

    volatilityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Rolling Volatility',
                data: volatilities,
                borderColor: '#f39c12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Volatility: ${formatPercentage(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function (value) {
                            return formatPercentage(value);
                        }
                    }
                }
            }
        }
    });
}

function displayExplanations(explanations) {
    const list = document.getElementById('explanations-list');
    list.innerHTML = '';

    if (!explanations || explanations.length === 0) {
        list.innerHTML = '<p class="explanation-item">No explanations available</p>';
        return;
    }

    // Show last 20 explanations
    const recentExplanations = explanations.slice(-20);

    recentExplanations.forEach(exp => {
        const item = document.createElement('div');
        item.className = 'explanation-item';
        item.innerHTML = `
            <div class="explanation-date">${exp.date || 'Unknown date'}</div>
            <div class="explanation-text">${exp.explanation || 'No explanation'}</div>
        `;
        list.appendChild(item);
    });
}

function displayRegimeTimeline(regimes) {
    const timeline = document.getElementById('regime-timeline');
    timeline.innerHTML = '';

    if (!regimes || regimes.length === 0) {
        timeline.innerHTML = '<p>No regime data available</p>';
        return;
    }

    // Sample regimes for display (show one per month)
    const sampleRate = Math.max(1, Math.floor(regimes.length / 20));
    const sampledRegimes = regimes.filter((_, i) => i % sampleRate === 0);

    sampledRegimes.forEach(regime => {
        const indicator = document.createElement('div');
        const regimeClass = (regime.regime || 'Normal').replace(/\s+/g, '-');
        indicator.className = `regime-indicator regime-${regimeClass}`;
        indicator.title = `${regime.date || 'Unknown'}: ${regime.regime || 'Normal'}\nVolatility: ${formatPercentage(regime.volatility || 0)}\nDrawdown: ${formatPercentage(regime.drawdown || 0)}`;
        indicator.textContent = regime.date ? regime.date.slice(5) : '??';
        timeline.appendChild(indicator);
    });
}

function formatCurrency(value) {
    if (value === undefined || value === null) return '$0';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

function formatPercentage(value) {
    if (value === undefined || value === null) return '0%';
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value);
}

// Make functions globally available for onclick handlers
window.addAsset = addAsset;
window.removeAsset = removeAsset;
window.createStressChart = createStressChart;
window.hideStressResults = hideStressResults;

// Function to create stacked volume chart
function createVolumeChart(volumes, dates) {
    const ctx = document.getElementById('volume-chart').getContext('2d');

    if (volumeChart) {
        volumeChart.destroy();
    }

    const datasets = Object.keys(volumes).map((asset, index) => {
        // Generate distinct colors
        const hue = (index * 137.508) % 360; // Golden angle approximation
        return {
            label: asset,
            data: volumes[asset],
            backgroundColor: `hsla(${hue}, 70%, 60%, 0.6)`,
            borderColor: `hsla(${hue}, 70%, 60%, 1)`,
            borderWidth: 1
        };
    });

    volumeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                    type: 'time',
                    time: {
                        unit: 'month'
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Volume'
                    },
                    ticks: {
                        callback: function (value) {
                            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                            if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                            return value;
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', { notation: "compact" }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

function createBacktestComparisonChart(rollingValues) {
    const ctx = document.getElementById('backtest-comparison-chart').getContext('2d');

    if (backtestComparisonChart) {
        backtestComparisonChart.destroy();
    }

    // Calculate returns for each window
    const dates = rollingValues.dates;

    // Helper to get return from value series or result object
    const getReturns = (seriesList) => {
        // If seriesList is object (from convert_to_python potentially), convert to array
        const list = Array.isArray(seriesList) ? seriesList : Object.values(seriesList);

        return list.map(item => {
            // Check if it's the new object structure with total_return
            if (item && typeof item.total_return === 'number') {
                return item.total_return;
            }

            // Fallback for array of values (legacy)
            if (Array.isArray(item) && item.length > 0) {
                const first = item[0];
                const last = item[item.length - 1];
                return (last - first) / first;
            }

            return 0;
        });
    };

    const returnsWithRisk = getReturns(rollingValues.with_risk);
    const returnsWithoutRisk = getReturns(rollingValues.without_risk);

    backtestComparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'With Risk Management',
                    data: returnsWithRisk,
                    backgroundColor: 'rgba(39, 174, 96, 0.7)',
                    borderColor: '#27ae60',
                    borderWidth: 1
                },
                {
                    label: 'Without Risk Management',
                    data: returnsWithoutRisk,
                    backgroundColor: 'rgba(231, 76, 60, 0.7)',
                    borderColor: '#e74c3c',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Window Return'
                    },
                    ticks: {
                        callback: function (value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Window End Date'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += (context.parsed.y * 100).toFixed(2) + '%';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}