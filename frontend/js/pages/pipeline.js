/**
 * DataForge — Pipeline Builder Page
 */

async function renderPipeline(container) {
  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header-title">⬡ Pipeline Builder</h1>
        <p class="page-header-desc">Build a data preprocessing pipeline step by step. Each transform is applied in order.</p>
      </div>

      ${!AppState.currentDatasetId ? `
        <div class="empty-state">
          <div class="empty-state-icon">◫</div>
          <div class="empty-state-title">No dataset loaded</div>
          <div class="empty-state-description">Go to the Data Explorer first and load a dataset. Then come back here to build your preprocessing pipeline.</div>
          <button class="btn btn-gradient mt-4" onclick="navigateTo('data')">Go to Data Explorer</button>
        </div>
      ` : `
        <div class="flex items-center gap-3 mb-6">
          <span style="font-size:20px">◉</span>
          <span style="font-weight:600">${AppState.currentDatasetName || 'Dataset'}</span>
          <span class="badge badge-primary">Active</span>
        </div>

        <!-- Pipeline Steps -->
        <div class="glass-card no-hover mb-6">
          <h3 class="heading-4 mb-4">Add Preprocessing Steps</h3>
          <p class="text-secondary mb-6" style="font-size:var(--text-sm)">Select transformations to apply to your data. Steps run top to bottom.</p>
          
          <div class="grid-3 mb-6">
            ${[
              { action: 'drop_missing', icon: '⊘', name: 'Drop Missing Rows', desc: 'Remove rows with any missing values' },
              { action: 'fill_missing_median', icon: '◉', name: 'Fill Missing (Median)', desc: 'Replace missing numbers with the median' },
              { action: 'fill_missing_mode', icon: '◈', name: 'Fill Missing (Mode)', desc: 'Replace missing with most common value' },
              { action: 'drop_duplicates', icon: '⊟', name: 'Drop Duplicates', desc: 'Remove duplicate rows' },
              { action: 'encode_label', icon: '⊞', name: 'Label Encode', desc: 'Convert text categories to numbers' },
              { action: 'scale_standard', icon: '≡', name: 'Standard Scale', desc: 'Normalize to zero mean, unit variance' },
              { action: 'scale_minmax', icon: '⇅', name: 'Min-Max Scale', desc: 'Scale values to 0-1 range' },
              { action: 'remove_outliers', icon: '⊗', name: 'Remove Outliers', desc: 'Remove extreme values using IQR method' },
            ].map(step => `
              <div class="model-card" onclick="addPipelineStep('${step.action}')" id="pipe-${step.action}">
                <div style="font-size:24px; margin-bottom:8px">${step.icon}</div>
                <div class="model-card-name">${step.name}</div>
                <div class="model-card-desc">${step.desc}</div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Current Pipeline -->
        <div class="glass-card no-hover mb-6">
          <h3 class="heading-4 mb-4">Current Pipeline</h3>
          <div id="pipeline-steps">
            <div class="text-center text-muted p-4" style="font-size:var(--text-sm)">
              No steps added yet. Click on the transformations above to build your pipeline.
            </div>
          </div>
        </div>

        <!-- Execute Button -->
        <div class="flex gap-3">
          <button class="btn btn-gradient btn-lg" onclick="executePipeline()" id="execute-pipeline-btn" disabled>
            Execute Pipeline
          </button>
          <button class="btn btn-outline btn-lg" onclick="clearPipeline()">
            Clear All
          </button>
        </div>

        <!-- Results -->
        <div id="pipeline-results" class="mt-6"></div>
      `}
    </div>
  `;
}

// Pipeline step storage
let pipelineSteps = [];

function addPipelineStep(action) {
  const names = {
    'drop_missing': 'Drop Missing Rows',
    'fill_missing_median': 'Fill Missing (Median)',
    'fill_missing_mode': 'Fill Missing (Mode)',
    'drop_duplicates': 'Drop Duplicates',
    'encode_label': 'Label Encode',
    'scale_standard': 'Standard Scale',
    'scale_minmax': 'Min-Max Scale',
    'remove_outliers': 'Remove Outliers',
  };

  pipelineSteps.push({ action, name: names[action] || action });
  updatePipelineUI();
  showToast(`Added: ${names[action]}`, 'success');
}

function removePipelineStep(index) {
  pipelineSteps.splice(index, 1);
  updatePipelineUI();
}

function clearPipeline() {
  pipelineSteps = [];
  updatePipelineUI();
}

function updatePipelineUI() {
  const stepsEl = document.getElementById('pipeline-steps');
  const executeBtn = document.getElementById('execute-pipeline-btn');
  
  if (!stepsEl) return;

  if (pipelineSteps.length === 0) {
    stepsEl.innerHTML = `<div class="text-center text-muted p-4" style="font-size:var(--text-sm)">No steps added yet.</div>`;
    if (executeBtn) executeBtn.disabled = true;
    return;
  }

  if (executeBtn) executeBtn.disabled = false;

  stepsEl.innerHTML = pipelineSteps.map((step, i) => `
    <div class="flex items-center gap-3 mb-2 animate-fade-in" style="padding:var(--space-3); background:var(--bg-surface); border-radius:var(--border-radius-md)">
      <span class="badge badge-primary" style="min-width:24px; text-align:center">${i + 1}</span>
      <span style="flex:1; font-weight:500; font-size:var(--text-sm)">${step.name}</span>
      <button class="btn btn-ghost btn-sm" onclick="removePipelineStep(${i})" style="color:var(--error)">✕</button>
    </div>
  `).join('');
}

async function executePipeline() {
  if (pipelineSteps.length === 0 || !AppState.currentDatasetId) return;

  const btn = document.getElementById('execute-pipeline-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const result = await API.executePipeline({
      dataset_id: AppState.currentDatasetId,
      steps: pipelineSteps.map(s => ({ action: s.action }))
    });

    // Update to transformed dataset
    AppState.currentDatasetId = result.dataset_id;
    AppState.datasets[result.dataset_id] = { stats: result.stats };

    document.getElementById('pipeline-results').innerHTML = `
      <div class="glass-card no-hover animate-fade-in-up" style="border-color:var(--success)">
        <h3 class="heading-4 mb-4" style="color:var(--success)">Pipeline Executed Successfully</h3>
        <div class="grid-3">
          <div class="stat-card">
            <div class="stat-card-label">Rows After</div>
            <div class="stat-card-value gradient">${result.rows.toLocaleString()}</div>
          </div>
          <div class="stat-card">
            <div class="stat-card-label">Columns After</div>
            <div class="stat-card-value gradient">${result.columns}</div>
          </div>
          <div class="stat-card">
            <div class="stat-card-label">Quality Score</div>
            <div class="stat-card-value" style="color:var(--success)">${result.stats.quality_score}/100</div>
          </div>
        </div>
        <div class="flex gap-3 mt-6">
          <button class="btn btn-primary" onclick="navigateTo('data')">View Transformed Data</button>
          <button class="btn btn-gradient" onclick="navigateTo('arena')">Train Models</button>
        </div>
      </div>
    `;

    showToast('Pipeline executed! Dataset transformed.', 'success');
  } catch (e) {
    showToast(`Pipeline error: ${e.message}`, 'error');
    document.getElementById('pipeline-results').innerHTML = `
      <div class="glass-card no-hover" style="border-color:var(--error)">
        <p style="color:var(--error)">${e.message}</p>
      </div>
    `;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}
