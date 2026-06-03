document.addEventListener('DOMContentLoaded', () => {
    // Inputs
    const tickerInput = document.getElementById('ticker');
    const startDateInput = document.getElementById('start_date');
    const endDateInput = document.getElementById('end_date');
    const episodesInput = document.getElementById('episodes');

    // Buttons
    const btnTrain = document.getElementById('btn-train');
    const btnBacktest = document.getElementById('btn-backtest');
    const btnChart = document.getElementById('btn-chart');
    const btnInfer = document.getElementById('btn-infer');

    // Status & Progress
    const statusText = document.getElementById('status-text');
    const statusBar = document.getElementById('status-bar');
    const progressContainer = document.getElementById('progress-container');
    const progressEp = document.getElementById('progress-ep');
    const progressReward = document.getElementById('progress-reward');
    const progressEps = document.getElementById('progress-eps');
    const progressFill = document.getElementById('progress-fill');

    // Panels
    const chartPanel = document.getElementById('chart-panel');
    const inferenceCard = document.getElementById('inference-card');
    const resultsContent = document.getElementById('results-content');

    let pollInterval = null;

    function getParams() {
        return {
            ticker: tickerInput.value.trim().toUpperCase() || 'AAPL',
            start_date: startDateInput.value,
            end_date: endDateInput.value,
            episodes: parseInt(episodesInput.value) || 8
        };
    }

    function setStatus(message, type = 'normal') {
        statusText.textContent = message;
        statusBar.className = 'status-bar ' + type;
    }

    function setButtonsState(disabled) {
        btnTrain.disabled = disabled;
        btnBacktest.disabled = disabled;
        btnChart.disabled = disabled;
        btnInfer.disabled = disabled;
    }

    // --- Action Execution (Train / Backtest) ---
    async function startAction(actionType) {
        const payload = getParams();
        const btn = actionType === 'train' ? btnTrain : btnBacktest;
        
        try {
            setButtonsState(true);
            btn.classList.add('loading');
            setStatus(`Initiating ${actionType} for ${payload.ticker}...`, 'running');

            if(actionType === 'train') progressContainer.style.display = 'block';

            const res = await fetch(`/api/${actionType}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            
            if(res.ok) {
                pollInterval = setInterval(() => checkStatus(actionType), 1000);
            } else {
                throw new Error(data.message || 'Unknown error');
            }
        } catch (err) {
            setStatus(`Error: ${err.message}`, 'error');
            btn.classList.remove('loading');
            setButtonsState(false);
        }
    }

    async function checkStatus(actionType) {
        try {
            const res = await fetch(`/api/${actionType}/status`);
            const data = await res.json();

            let type = 'normal';
            if(data.status === 'error') type = 'error';
            if(data.status === 'running') type = 'running';
            if(data.status === 'completed') type = 'success';

            setStatus(data.message || `Running ${actionType}...`, type);

            // Update Progress Bar
            if(actionType === 'train' && data.episodes > 0) {
                const pct = (data.progress / data.episodes) * 100;
                progressFill.style.width = `${pct}%`;
                progressEp.textContent = `Episode: ${data.progress}/${data.episodes}`;
                progressReward.textContent = `Reward: ${data.reward}`;
                progressEps.textContent = `Epsilon: ${data.epsilon}`;
            }

            if(data.status === 'completed' || data.status === 'error') {
                clearInterval(pollInterval);
                const btn = actionType === 'train' ? btnTrain : btnBacktest;
                btn.classList.remove('loading');
                setButtonsState(false);

                if(data.status === 'completed') {
                    showResults();
                    if(actionType === 'train') {
                        setTimeout(() => { progressContainer.style.display = 'none'; }, 3000);
                    }
                }
            }
        } catch (err) {
            console.error(err);
        }
    }

    function showResults() {
        resultsContent.classList.remove('empty');
        const ts = new Date().getTime();
        resultsContent.innerHTML = `
            <img src="/results/learning_curve.png?t=${ts}" class="result-image" alt="Learning Curve" onerror="this.style.display='none'">
            <img src="/results/backtest_results.png?t=${ts}" class="result-image" alt="Backtest Results" onerror="this.style.display='none'">
        `;
    }

    // --- Chart Data ---
    async function loadChart() {
        try {
            setButtonsState(true);
            btnChart.classList.add('loading');
            setStatus('Loading chart data...', 'running');

            const res = await fetch('/api/data/chart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(getParams())
            });
            const data = await res.json();

            if(!res.ok) throw new Error(data.message);

            chartPanel.style.display = 'block';
            
            const trace = {
                x: data.dates,
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
                type: 'candlestick',
                xaxis: 'x',
                yaxis: 'y'
            };

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e2e8f0', family: 'Outfit' },
                margin: { t: 20, r: 20, l: 40, b: 40 },
                xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.1)' }
            };

            Plotly.newPlot('candlestick-chart', [trace], layout, {responsive: true});
            setStatus('Chart loaded.', 'success');
        } catch(err) {
            setStatus(`Chart Error: ${err.message}`, 'error');
        } finally {
            btnChart.classList.remove('loading');
            setButtonsState(false);
        }
    }

    // --- Inference ---
    async function predictLatest() {
        try {
            setButtonsState(true);
            btnInfer.classList.add('loading');
            setStatus('Predicting latest action...', 'running');

            const res = await fetch('/api/inference', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(getParams())
            });
            const data = await res.json();

            if(!res.ok) throw new Error(data.message);

            inferenceCard.style.display = 'block';
            
            const badge = document.getElementById('inf-action');
            badge.textContent = data.action;
            badge.className = 'action-badge ' + data.action.toLowerCase();
            
            document.getElementById('inf-explanation').textContent = data.explanation;
            
            document.getElementById('q-sell').textContent = data.q_values[0].toFixed(3);
            document.getElementById('q-hold').textContent = data.q_values[1].toFixed(3);
            document.getElementById('q-buy').textContent = data.q_values[2].toFixed(3);

            setStatus('Prediction complete.', 'success');
        } catch(err) {
            setStatus(`Inference Error: ${err.message}`, 'error');
        } finally {
            btnInfer.classList.remove('loading');
            setButtonsState(false);
        }
    }

    // Event Listeners
    btnTrain.addEventListener('click', () => startAction('train'));
    btnBacktest.addEventListener('click', () => startAction('backtest'));
    btnChart.addEventListener('click', loadChart);
    btnInfer.addEventListener('click', predictLatest);
});
