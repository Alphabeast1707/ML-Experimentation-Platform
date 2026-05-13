/**
 * DataForge — Model Arena Page
 */

async function renderArena(container) {
  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header-title">△ Model Arena</h1>
        <p class="page-header-desc">Train multiple models and battle them head-to-head. The best model wins the crown!</p>
      </div>

      ${!AppState.currentDatasetId ? `
        <div class="empty-state">
          <div class="empty-state-icon">◫</div>
          <div class="empty-state-title">No dataset loaded</div>
          <div class="empty-state-description">Load a dataset in the Data Explorer first, then come back here to train and compare models.</div>
          <button class="btn btn-gradient mt-4" onclick="navigateTo('data')">Go to Data Explorer</button>
        </div>
      ` : `
        <!-- Configuration -->
        <div class="glass-card no-hover mb-6">
          <h3 class="heading-4 mb-2">Configure Your Battle</h3>
          <p class="text-secondary mb-6" style="font-size:var(--text-sm)">Choose what to predict, select your models, and let them fight!</p>
          
          <div class="form-row mb-6">
            <div class="form-group">
              <label class="form-label">Dataset</label>
              <div class="flex items-center gap-2">
                <span style="font-size:20px">◉</span>
                <span style="font-weight:600">${AppState.currentDatasetName || 'Current Dataset'}</span>
                <span class="badge badge-success">Loaded</span>
              </div>
            </div>
            <div class="form-group">
              <label class="form-label">Target Column (what to predict)</label>
              <select class="form-select" id="arena-target" onchange="updateArenaFeatures()">
                <option value="">Loading columns...</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">Problem Type</label>
              <select class="form-select" id="arena-problem-type">
                <option value="classification">Classification (categories)</option>
                <option value="regression">Regression (numbers)</option>
              </select>
            </div>
          </div>

          <!-- Feature Selection -->
          <div class="form-group mb-6">
            <label class="form-label">Features to Use <span class="text-muted">(leave unchecked to use all except target)</span></label>
            <div class="checkbox-group" id="arena-features">
              <span class="text-muted" style="font-size:var(--text-sm)">Select a target column first</span>
            </div>
          </div>

          <!-- Model Selection -->
          <div class="form-group mb-6">
            <label class="form-label">Select Models to Battle</label>
            <p class="text-muted mb-3" style="font-size:var(--text-xs)">Pick 2 or more models. More models = more fun!</p>
            <div class="grid-3" id="arena-models">
              ${generateModelCards()}
            </div>
          </div>

          <!-- Test Size -->
          <div class="form-row mb-6">
            <div class="form-group">
              <label class="form-label">Test Split Size</label>
              <select class="form-select" id="arena-test-size">
                <option value="0.1">10% test</option>
                <option value="0.2" selected>20% test (recommended)</option>
                <option value="0.3">30% test</option>
              </select>
            </div>
          </div>

          <!-- Battle Button -->
          <button class="btn btn-gradient btn-lg" onclick="startBattle()" id="battle-btn">
            Start Battle
          </button>
        </div>

        <!-- Results -->
        <div id="arena-results"></div>
      `}
    </div>
  `;

  // Load columns if dataset is active
  if (AppState.currentDatasetId) {
    try {
      const data = await API.getColumns(AppState.currentDatasetId);
      const select = document.getElementById('arena-target');
      if (select) {
        select.innerHTML = '<option value="">Select target column...</option>' +
          data.columns.map(c => `<option value="${c.name}">${c.name} (${c.is_numeric ? 'numeric' : 'categorical'} • ${c.unique} unique)</option>`).join('');
      }
    } catch (e) {
      showToast('Failed to load columns', 'error');
    }
  }
}

function generateModelCards() {
  const models = [
    { id: 'random_forest', icon: '◣', name: 'Random Forest', desc: 'Reliable all-rounder. Great default.', recommended: true },
    { id: 'logistic_regression', icon: '∕', name: 'Logistic Regression', desc: 'Simple and interpretable baseline.' },
    { id: 'decision_tree', icon: '◷', name: 'Decision Tree', desc: 'Easy to understand. Good for learning.' },
    { id: 'gradient_boosting', icon: '▲', name: 'Gradient Boosting', desc: 'Powerful. Builds trees sequentially.' },
    { id: 'svm', icon: '◎', name: 'SVM', desc: 'Great for complex boundaries.' },
    { id: 'knn', icon: '○', name: 'K-Nearest Neighbors', desc: 'Classifies by nearest examples.' },
    { id: 'naive_bayes', icon: '▽', name: 'Naive Bayes', desc: 'Fast and surprisingly effective.' },
    { id: 'adaboost', icon: '▶', name: 'AdaBoost', desc: 'Focuses on hard examples.' },
    { id: 'xgboost', icon: '⚡', name: 'XGBoost', desc: 'State-of-the-art. Competition winner.' },
    { id: 'lightgbm', icon: '◆', name: 'LightGBM', desc: 'Fast boosting for large data.' },
  ];

  return models.map(m => `
    <div class="model-card" id="model-card-${m.id}" onclick="toggleModel('${m.id}')">
      <div style="font-size:24px; margin-bottom:4px">${m.icon}</div>
      <div class="model-card-name">${m.name} ${m.recommended ? '<span class="badge badge-success" style="font-size:9px">Recommended</span>' : ''}</div>
      <div class="model-card-desc">${m.desc}</div>
    </div>
  `).join('');
}

let selectedModels = new Set();

function toggleModel(modelId) {
  const card = document.getElementById(`model-card-${modelId}`);
  if (selectedModels.has(modelId)) {
    selectedModels.delete(modelId);
    card.classList.remove('selected');
  } else {
    selectedModels.add(modelId);
    card.classList.add('selected');
  }
}

function updateArenaFeatures() {
  const target = document.getElementById('arena-target')?.value;
  const featuresEl = document.getElementById('arena-features');
  if (!target || !featuresEl) return;

  const ds = AppState.datasets[AppState.currentDatasetId];
  if (!ds || !ds.stats) return;

  const columns = ds.stats.column_names.filter(c => c !== target);
  featuresEl.innerHTML = columns.map(col => `
    <label class="checkbox-item">
      <input type="checkbox" value="${col}" checked>
      ${col}
    </label>
  `).join('');

  // Auto-detect problem type
  const select = document.getElementById('arena-target');
  const selectedOption = select.options[select.selectedIndex];
  if (selectedOption && selectedOption.text.includes('categorical')) {
    document.getElementById('arena-problem-type').value = 'classification';
  }
}

async function startBattle() {
  const target = document.getElementById('arena-target')?.value;
  const problemType = document.getElementById('arena-problem-type')?.value;
  const testSize = parseFloat(document.getElementById('arena-test-size')?.value || '0.2');

  if (!target) {
    showToast('Please select a target column!', 'warning');
    return;
  }
  if (selectedModels.size < 2) {
    showToast('Select at least 2 models to battle!', 'warning');
    return;
  }

  // Get selected features
  const featureCheckboxes = document.querySelectorAll('#arena-features input[type="checkbox"]:checked');
  const features = Array.from(featureCheckboxes).map(cb => cb.value);

  const btn = document.getElementById('battle-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  const resultsEl = document.getElementById('arena-results');
  resultsEl.innerHTML = `
    <div class="glass-card no-hover text-center p-6 mt-6">
      <div style="font-size:48px; margin-bottom:16px" class="animate-float">△</div>
      <h3 class="heading-4 mb-2">Battle in Progress...</h3>
      <p class="text-secondary">Training ${selectedModels.size} models and comparing performance. This may take a moment.</p>
      <div class="loading-dots mt-4"><span></span><span></span><span></span></div>
    </div>
  `;

  try {
    const result = await API.compareModels({
      dataset_id: AppState.currentDatasetId,
      model_types: Array.from(selectedModels),
      target_column: target,
      feature_columns: features.length > 0 ? features : undefined,
      problem_type: problemType,
      test_size: testSize
    });

    renderBattleResults(result.results, problemType);
    showToast(`Battle complete! ${result.results.length} models trained.`, 'success');
  } catch (e) {
    resultsEl.innerHTML = `
      <div class="glass-card no-hover mt-6" style="border-color:var(--error)">
        <p style="color:var(--error)">Battle failed: ${e.message}</p>
        <p class="text-secondary mt-2" style="font-size:var(--text-sm)">Common fixes: Make sure your target column and features are compatible. Try encoding categorical features first in Pipeline Builder.</p>
      </div>
    `;
    showToast(`Battle error: ${e.message}`, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function renderBattleResults(results, problemType) {
  const resultsEl = document.getElementById('arena-results');
  if (!results || results.length === 0) {
    resultsEl.innerHTML = '<div class="text-muted">No results</div>';
    return;
  }

  const winner = results[0];
  const primaryMetric = problemType === 'classification' ? 'accuracy' : 'r2';
  const metricLabel = problemType === 'classification' ? 'Accuracy' : 'R² Score';

  resultsEl.innerHTML = `
    <!-- Winner Banner -->
    <div class="glass-card no-hover mb-6 mt-6 animate-scale-in" style="border-color:var(--success); background:linear-gradient(135deg, rgba(16,185,129,0.05), rgba(6,182,212,0.05))">
      <div class="text-center">
        <div style="font-size:56px; margin-bottom:8px">★</div>
        <h3 class="heading-3 mb-2">Winner: <span class="text-gradient">${winner.model_name || winner.model_type}</span></h3>
        <p class="text-secondary mb-4">${metricLabel}: <span style="font-size:var(--text-2xl); font-weight:800; color:var(--success)">${((winner.metrics[primaryMetric] || 0) * 100).toFixed(1)}%</span></p>
        ${winner.metrics.training_time ? `<span class="badge badge-info">Trained in ${winner.metrics.training_time}s</span>` : ''}
      </div>
    </div>

    <!-- Leaderboard -->
    <div class="glass-card no-hover mb-6 animate-fade-in-up">
      <h3 class="heading-4 mb-4">🏅 Leaderboard</h3>
      <div>
        ${results.map((r, i) => {
          const rank = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : 'normal';
          const metricVal = r.metrics[primaryMetric] || 0;
          return `
            <div class="leaderboard-item">
              <div class="leaderboard-rank ${rank}">${i + 1}</div>
              <div style="flex:1">
                <div style="font-weight:600">${r.model_name || r.model_type}</div>
                <div class="text-muted" style="font-size:var(--text-xs)">${r.error ? `Error: ${r.error}` : `Trained in ${r.metrics.training_time || '?'}s`}</div>
              </div>
              ${!r.error ? `
                <div class="flex gap-4" style="font-family:var(--font-mono); font-size:var(--text-sm)">
                  ${problemType === 'classification' ? `
                    <div class="text-center">
                      <div class="text-muted" style="font-size:10px; text-transform:uppercase">Accuracy</div>
                      <div style="color:${metricColor(r.metrics.accuracy || 0)}; font-weight:700">${((r.metrics.accuracy || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="text-center">
                      <div class="text-muted" style="font-size:10px; text-transform:uppercase">F1</div>
                      <div style="color:${metricColor(r.metrics.f1 || 0)}; font-weight:700">${((r.metrics.f1 || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="text-center">
                      <div class="text-muted" style="font-size:10px; text-transform:uppercase">Precision</div>
                      <div>${((r.metrics.precision || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="text-center">
                      <div class="text-muted" style="font-size:10px; text-transform:uppercase">Recall</div>
                      <div>${((r.metrics.recall || 0) * 100).toFixed(1)}%</div>
                    </div>
                  ` : `
                    <div class="text-center">
                      <div class="text-muted" style="font-size:10px; text-transform:uppercase">R²</div>
                      <div style="color:${metricColor(r.metrics.r2 || 0)}; font-weight:700">${(r.metrics.r2 || 0).toFixed(4)}</div>
                    </div>
                    <div class="text-center">
                      <div class="text-muted" style="font-size:10px; text-transform:uppercase">RMSE</div>
                      <div>${(r.metrics.rmse || 0).toFixed(4)}</div>
                    </div>
                    <div class="text-center">
                      <div class="text-muted" style="font-size:10px; text-transform:uppercase">MAE</div>
                      <div>${(r.metrics.mae || 0).toFixed(4)}</div>
                    </div>
                  `}
                </div>
              ` : ''}
            </div>
          `;
        }).join('')}
      </div>
    </div>

    <!-- Detailed Metrics Comparison -->
    <div class="glass-card no-hover mb-6 animate-fade-in-up">
      <h3 class="heading-4 mb-4">Performance Comparison</h3>
      ${generateBarChart(
        results.filter(r => !r.error).map(r => ({
          label: r.model_name || r.model_type,
          value: (r.metrics[primaryMetric] || 0)
        })),
        1.0,
        'var(--gradient-hero)'
      )}
    </div>

    <!-- Feature Importance (from winner) -->
    ${winner.feature_importance && winner.feature_importance.length > 0 ? `
      <div class="glass-card no-hover mb-6 animate-fade-in-up">
        <h3 class="heading-4 mb-4">Feature Importance <span class="text-muted" style="font-size:var(--text-sm)">(from ${winner.model_name || winner.model_type})</span></h3>
        <p class="text-secondary mb-4" style="font-size:var(--text-sm)">Which features matter most for predictions. Higher = more important.</p>
        ${generateBarChart(
          winner.feature_importance.slice(0, 10).map(f => ({ label: f.feature, value: f.importance })),
          null,
          'var(--accent)'
        )}
      </div>
    ` : ''}

    <!-- Confusion Matrix (winner, classification only) -->
    ${winner.confusion_matrix && problemType === 'classification' ? `
      <div class="glass-card no-hover mb-6 animate-fade-in-up">
        <h3 class="heading-4 mb-4">Confusion Matrix <span class="text-muted" style="font-size:var(--text-sm)">(${winner.model_name || winner.model_type})</span></h3>
        <p class="text-secondary mb-4" style="font-size:var(--text-sm)">Diagonal values (highlighted) are correct predictions. Off-diagonal are errors.</p>
        <div class="flex justify-center">
          ${generateConfusionMatrix(winner.confusion_matrix)}
        </div>
      </div>
    ` : ''}

    <!-- Sample Predictions -->
    ${winner.predictions_sample && winner.predictions_sample.length > 0 ? `
      <div class="glass-card no-hover animate-fade-in-up">
        <h3 class="heading-4 mb-4">Sample Predictions <span class="text-muted" style="font-size:var(--text-sm)">(${winner.model_name || winner.model_type})</span></h3>
        <div class="data-table-wrapper">
          <table class="data-table">
            <thead><tr><th>#</th><th>Actual</th><th>Predicted</th><th>Result</th></tr></thead>
            <tbody>
              ${winner.predictions_sample.map((p, i) => {
                const match = String(p.actual) === String(p.predicted);
                return `<tr>
                  <td>${i + 1}</td>
                  <td>${p.actual}</td>
                  <td>${p.predicted}</td>
                  <td>${match ? '<span style="color:var(--success)">Correct</span>' : '<span style="color:var(--error)">Wrong</span>'}</td>
                </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    ` : ''}
  `;
}
