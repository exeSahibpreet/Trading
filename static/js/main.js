let currentMode = 'single-tab';

const tabTitleMap = {
    'single-tab': 'Backtest one strategy in isolation',
    'ranking-tab': 'Rank the five active strategies on out-of-sample performance',
    'regime-tab': 'Run the adaptive engine that switches strategies by forecasted regime',
    'walkforward-tab': 'Review fold-by-fold walk-forward profits and parameter adaptation'
};

function showError(message) {
    const banner = document.getElementById('error-banner');
    banner.textContent = message;
    banner.classList.remove('hidden');
}

function clearError() {
    const banner = document.getElementById('error-banner');
    banner.textContent = '';
    banner.classList.add('hidden');
}

function getApiUrl(endpoint) {
    return new URL(endpoint.replace(/^\//, ''), window.location.href).toString();
}

async function apiRequest(endpoint, payload) {
    const url = getApiUrl(endpoint);
    console.log(`[request] POST ${url}`, payload);

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    const rawText = await response.text();
    console.log(`[response] ${response.status} ${url}`);

    let data;
    try {
        data = rawText ? JSON.parse(rawText) : {};
    } catch (parseError) {
        console.error('[response] Non-JSON payload received', rawText.slice(0, 500));
        throw new Error(
            `Server returned ${response.status} ${response.statusText}. ` +
            `Expected JSON but received something else, often an HTML error/proxy page.`
        );
    }

    if (!response.ok) {
        console.error('[response] API error', data);
        throw new Error(data.error || `Request failed with status ${response.status}.`);
    }

    return data;
}

function toggleViews(activeViewId = null) {
    ['single-view', 'ranking-view', 'regime-view', 'walkforward-view'].forEach((id) => {
        document.getElementById(id).classList.toggle('hidden', id !== activeViewId);
    });
}

document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', (event) => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        event.target.classList.add('active');
        currentMode = event.target.dataset.tab;

        document.getElementById('page-title').textContent = tabTitleMap[currentMode];
        document.getElementById('strategy-block').classList.toggle('hidden', currentMode !== 'single-tab');
        document.getElementById('regime-toggles').classList.toggle('hidden', currentMode !== 'regime-tab');
        document.getElementById('walkforward-controls').classList.toggle('hidden', currentMode !== 'walkforward-tab');

        if (currentMode === 'single-tab') {
            document.getElementById('run-btn').textContent = 'Run Selected Strategy';
            document.getElementById('info-note').innerHTML = '<p class="info-title">Model notes</p><p>Single-strategy mode helps validate one idea before letting the adaptive router switch between them.</p>';
        } else if (currentMode === 'ranking-tab') {
            document.getElementById('run-btn').textContent = 'Rank All Strategies';
            document.getElementById('info-note').innerHTML = '<p class="info-title">Ranking notes</p><p>Scores balance Calmar, Sharpe, walk-forward efficiency, drawdown, recovery, win rate, and sample size.</p>';
        } else if (currentMode === 'regime-tab') {
            document.getElementById('run-btn').textContent = 'Run Adaptive Engine';
            document.getElementById('info-note').innerHTML = '<p class="info-title">Adaptive notes</p><p>The engine forecasts the next-bar regime, turns on the matching strategy, and keeps a smaller fallback allocation when confidence is weak.</p>';
        } else {
            document.getElementById('run-btn').textContent = 'Run Walk-Forward';
            document.getElementById('info-note').innerHTML = '<p class="info-title">Walk-forward notes</p><p>This mode shows each out-of-sample test fold separately, so you can see how much profit was earned in every testing stage.</p>';
        }

        clearError();
        toggleViews(null);
    });
});

document.getElementById('backtest-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    toggleViews(null);
    document.getElementById('loading').classList.remove('hidden');

    const formData = new FormData(event.target);
    const index = formData.get('index');
    const mode = formData.get('mode');

    try {
        if (currentMode === 'single-tab') {
            const strategy = formData.get('strategy');
            const data = await apiRequest('api/run_backtest', { strategy, index, mode });
            renderSingleResults(data);
            toggleViews('single-view');
        } else if (currentMode === 'ranking-tab') {
            const data = await apiRequest('api/rank_strategies', { index, mode });
            renderRankingResults(data.ranked || []);
            toggleViews('ranking-view');
        } else if (currentMode === 'regime-tab') {
            const toggles = {};
            Object.keys(window.strategyLabels).forEach((strategyKey) => {
                toggles[strategyKey] = formData.get(`en_${strategyKey}`) === 'on';
            });

            const data = await apiRequest('api/run_portfolio', { index, mode, toggles });
            renderRegimeResults(data);
            toggleViews('regime-view');
        } else {
            const anchored = formData.get('anchored_walkforward') === 'on';
            const data = await apiRequest('api/run_walk_forward', { index, mode, anchored });
            renderWalkForwardResults(data);
            toggleViews('walkforward-view');
        }
    } catch (error) {
        console.error('[ui] Request failed', error);
        showError(error.message || 'Something went wrong.');
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
});

function formatCurrency(value) {
    return `Rs ${Math.round(value).toLocaleString('en-IN')}`;
}

function formatPct(value) {
    return `${Number(value || 0).toFixed(2)}%`;
}

function setColor(element, value) {
    if (!element) return;
    element.classList.remove('positive', 'negative');
    if (value > 0) element.classList.add('positive');
    if (value < 0) element.classList.add('negative');
}

function baseLayout() {
    return {
        margin: { t: 16, r: 20, l: 56, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'IBM Plex Sans, sans-serif', color: '#10233e' },
        xaxis: { showgrid: true, gridcolor: 'rgba(16, 35, 62, 0.08)' },
        yaxis: { showgrid: true, gridcolor: 'rgba(16, 35, 62, 0.08)' },
        hovermode: 'x unified'
    };
}

function renderSingleResults(data) {
    const train = data.train.metrics;
    const test = data.test.metrics;
    const wfe = (train.annual_return || 0) > 0 ? Math.max(0, (test.annual_return || 0) / train.annual_return) : 0;

    document.getElementById('test-pnl').textContent = `${formatCurrency(test.total_profit)} | ${formatCurrency(test.final_capital)}`;
    document.getElementById('test-sharpe').textContent = `${(test.sharpe_ratio || 0).toFixed(2)} | ${(test.sortino_ratio || 0).toFixed(2)}`;
    document.getElementById('test-calmar').textContent = `${(test.calmar_ratio || 0).toFixed(2)} | ${wfe.toFixed(2)}`;
    document.getElementById('test-dd').textContent = `${formatPct(test.max_drawdown)} | ${test.recovery_days || 0}d`;
    document.getElementById('test-winrate').textContent = `${formatPct(test.win_rate)} | ${test.num_trades || 0}`;
    document.getElementById('test-rec-fac').textContent = (test.recovery_factor || 0).toFixed(2);
    setColor(document.getElementById('test-pnl'), test.total_profit || 0);

    const dashboard = document.getElementById('health-dashboard');
    dashboard.innerHTML = '';
    const badges = [];
    if ((test.num_trades || 0) < 30) badges.push(['Thin sample', 'danger']);
    if ((test.sharpe_ratio || 0) < 0) badges.push(['Negative Sharpe', 'danger']);
    else if ((test.sharpe_ratio || 0) < 1) badges.push(['Risk-adjusted weak', 'warning']);
    if (wfe < 0.5) badges.push(['Overfit risk', 'danger']);
    if ((test.max_drawdown || 0) < -30) badges.push(['Deep drawdown', 'danger']);
    if ((train.total_profit || 0) > 0 && (test.total_profit || 0) < 0) badges.push(['Failed OOS', 'danger']);
    if (!badges.length) badges.push(['Healthy profile', 'success']);
    badges.forEach(([label, kind]) => {
        dashboard.innerHTML += `<span class="badge badge-${kind}">${label}</span>`;
    });

    plotSingleChart('train-chart', data.train.dates, data.train.equity, train.drawdown_series || [], 'Train');
    plotSingleChart('test-chart', data.test.dates, data.test.equity, test.drawdown_series || [], 'Test');
}

function plotSingleChart(containerId, dates, equity, drawdown, seriesName) {
    const layout = baseLayout();
    layout.grid = { rows: 2, columns: 1, pattern: 'independent', ygap: 0.1 };
    layout.height = 380;
    layout.showlegend = false;
    layout.yaxis = { title: 'Equity', domain: [0.35, 1], showgrid: true, gridcolor: 'rgba(16, 35, 62, 0.08)' };
    layout.yaxis2 = { title: 'Drawdown %', domain: [0, 0.22], showgrid: false };

    Plotly.newPlot(containerId, [
        {
            x: dates,
            y: equity,
            type: 'scatter',
            mode: 'lines',
            name: `${seriesName} Equity`,
            line: { color: '#0f8b8d', width: 3 },
            fill: 'tozeroy',
            fillcolor: 'rgba(15, 139, 141, 0.10)'
        },
        {
            x: dates,
            y: drawdown,
            type: 'scatter',
            mode: 'lines',
            name: `${seriesName} DD`,
            line: { color: '#d64550', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(214, 69, 80, 0.12)',
            yaxis: 'y2'
        }
    ], layout, { responsive: true, displayModeBar: false });
}

function renderRankingResults(ranked) {
    const tbody = document.querySelector('#ranking-table tbody');
    tbody.innerHTML = '';

    const colors = ['#0f8b8d', '#f4a261', '#d64550', '#2a9d8f', '#3d5a80'];
    const traces = [];

    ranked.forEach((row, index) => {
        const metrics = row.test_metrics || {};
        const tr = document.createElement('tr');
        if (index === 0) tr.classList.add('top-rank');
        if (row.is_disqualified) tr.classList.add('muted-row');

        let flagHtml = '';
        (row.badges || []).forEach((badge) => {
            flagHtml += `<span class="badge badge-danger compact">${badge}</span>`;
        });
        if (!flagHtml) flagHtml = '<span class="badge badge-success compact">Clean</span>';

        tr.innerHTML = `
            <td>#${index + 1}</td>
            <td><strong>${row.strategy_name}</strong></td>
            <td><span class="score-badge">${Number(row.score || 0).toFixed(1)}</span></td>
            <td>${formatCurrency(metrics.final_capital || 0)}</td>
            <td>${metrics.num_trades || 0}</td>
            <td>${Number(metrics.calmar_ratio || 0).toFixed(2)} | ${Number(metrics.sharpe_ratio || 0).toFixed(2)}</td>
            <td class="${(metrics.max_drawdown || 0) >= -20 ? 'positive' : 'negative'}">${formatPct(metrics.max_drawdown || 0)}</td>
            <td>${Number(row.wfe || 0).toFixed(2)}</td>
            <td>${flagHtml}</td>
        `;
        tbody.appendChild(tr);

        traces.push({
            x: row.dates || [],
            y: row.equity || [],
            mode: 'lines',
            name: row.strategy_name,
            line: { width: index === 0 ? 3.5 : 2, color: colors[index % colors.length] }
        });
    });

    const layout = baseLayout();
    layout.height = 430;
    layout.showlegend = true;
    layout.legend = { orientation: 'h', y: -0.2 };
    Plotly.newPlot('compare-chart', traces, layout, { responsive: true, displayModeBar: false });
}

function getRegimeColor(regime) {
    const map = {
        'TRENDING': 'rgba(42, 157, 143, 0.16)',
        'RANGE_BOUND': 'rgba(244, 162, 97, 0.16)',
        'BREAKOUT_EXPANSION': 'rgba(61, 90, 128, 0.16)',
        'EVENT_DRIVEN': 'rgba(214, 69, 80, 0.16)',
        'INSTITUTIONAL_FLOW': 'rgba(131, 56, 236, 0.14)',
        'BALANCED': 'rgba(16, 35, 62, 0.08)',
        'INSUFFICIENT_DATA': 'rgba(120, 120, 120, 0.08)'
    };
    return map[regime] || 'rgba(16, 35, 62, 0.08)';
}

function renderRegimeResults(data) {
    const test = data.test || {};
    const metrics = test.combined_metrics || {};

    document.getElementById('port-pnl').textContent = `${formatCurrency(metrics.total_profit || 0)} | ${formatCurrency(metrics.final_capital || 0)}`;
    document.getElementById('port-sharpe').textContent = Number(metrics.sharpe_ratio || 0).toFixed(2);
    document.getElementById('port-calmar').textContent = Number(metrics.calmar_ratio || 0).toFixed(2);
    document.getElementById('port-dd').textContent = formatPct(metrics.max_drawdown || 0);
    setColor(document.getElementById('port-pnl'), metrics.total_profit || 0);
    setColor(document.getElementById('port-dd'), (metrics.max_drawdown || 0) >= -20 ? 1 : -1);

    const dates = test.dates || [];
    const regimes = test.regimes || [];
    const shapes = [];
    let currentRegime = regimes[0];
    let startIndex = 0;

    for (let i = 1; i <= regimes.length; i += 1) {
        if (i === regimes.length || regimes[i] !== currentRegime) {
            if (dates[startIndex] && dates[i - 1]) {
                shapes.push({
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0: dates[startIndex],
                    x1: dates[i - 1],
                    y0: 0,
                    y1: 1,
                    fillcolor: getRegimeColor(currentRegime),
                    line: { width: 0 },
                    layer: 'below'
                });
            }
            currentRegime = regimes[i];
            startIndex = i;
        }
    }

    const equityLayout = baseLayout();
    equityLayout.height = 430;
    equityLayout.shapes = shapes;
    Plotly.newPlot('port-equity-chart', [{
        x: dates,
        y: test.combined_equity || [],
        type: 'scatter',
        mode: 'lines',
        name: 'Adaptive Equity',
        line: { color: '#10233e', width: 3 }
    }], equityLayout, { responsive: true, displayModeBar: false });

    const stratEntries = Object.entries(test.strat_equities || {});
    const strategyLayout = baseLayout();
    strategyLayout.height = 430;
    strategyLayout.showlegend = true;
    strategyLayout.legend = { orientation: 'h', y: -0.22 };
    const strategyPalette = ['#0f8b8d', '#f4a261', '#d64550', '#3d5a80', '#8338ec'];
    Plotly.newPlot('port-strat-chart', stratEntries.map(([key, values], index) => ({
        x: dates,
        y: values,
        mode: 'lines',
        name: (test.strategy_labels || window.strategyLabels)[key] || key,
        line: { width: 2, color: strategyPalette[index % strategyPalette.length] }
    })), strategyLayout, { responsive: true, displayModeBar: false });

    const regimeBody = document.querySelector('#regime-pnl-table tbody');
    regimeBody.innerHTML = '';
    Object.entries(test.regime_pnl || {}).forEach(([regime, pnl]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td><strong>${regime.replaceAll('_', ' ')}</strong></td><td class="${pnl >= 0 ? 'positive' : 'negative'}">${Number(pnl).toFixed(2)}%</td>`;
        regimeBody.appendChild(tr);
    });

    const rotationBody = document.querySelector('#strategy-rotation-table tbody');
    rotationBody.innerHTML = '';
    const counts = {};
    (test.recommended_strategies || []).forEach((strategy) => {
        counts[strategy] = (counts[strategy] || 0) + 1;
    });
    Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([strategy, count]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td><strong>${(test.strategy_labels || window.strategyLabels)[strategy] || strategy}</strong></td><td>${count}</td>`;
        rotationBody.appendChild(tr);
    });
}

function renderWalkForwardResults(data) {
    const rows = data.summary || [];
    document.getElementById('wf-final-wfe').textContent = Number(data.final_wfe || 0).toFixed(2);
    document.getElementById('wf-overfit').textContent = data.overfit_flag ? 'Yes' : 'No';
    document.getElementById('wf-fold-count').textContent = rows.length;

    const totalProfit = rows.reduce((sum, row) => sum + Number(row.test_profit || 0), 0);
    document.getElementById('wf-total-profit').textContent = formatCurrency(totalProfit);
    setColor(document.getElementById('wf-total-profit'), totalProfit);
    setColor(document.getElementById('wf-overfit'), data.overfit_flag ? -1 : 1);

    const tbody = document.querySelector('#walkforward-table tbody');
    tbody.innerHTML = '';

    rows.forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>#${row.fold}</td>
            <td>${row.train_label || '-'}</td>
            <td>${row.test_label || '-'}</td>
            <td class="${Number(row.test_profit || 0) >= 0 ? 'positive' : 'negative'}">${formatCurrency(row.test_profit || 0)}</td>
            <td>${Number(row.test_profit_factor || 0).toFixed(2)}</td>
            <td>${String(row.strategy_dominance || '-').replaceAll('_', ' ')}</td>
            <td>${Number(row.wfe || 0).toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
    });
}
