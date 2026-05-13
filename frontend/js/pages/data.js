/**
 * DataForge — Data Explorer Page
 */

async function renderDataExplorer(container) {
  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header-title">◫ Data Explorer</h1>
        <p class="page-header-desc">Upload your dataset or pick a sample. We'll automatically analyze everything for you.</p>
      </div>

      <!-- Upload / Sample Selection -->
      <div class="grid-2 mb-8" style="grid-template-columns: 1fr 1fr;">
        <!-- Upload -->
        <div class="glass-card no-hover">
          <h3 class="heading-4 mb-4">Upload Your Data</h3>
          <div class="dropzone" id="dropzone"
               ondragover="event.preventDefault(); this.classList.add('drag-over')"
               ondragleave="this.classList.remove('drag-over')"
               ondrop="handleFileDrop(event)"
               onclick="document.getElementById('file-input').click()">
            <div class="dropzone-icon">📁</div>
            <div class="dropzone-text">Drag & drop your CSV file here</div>
            <div class="dropzone-hint">or click to browse • CSV, Excel supported</div>
          </div>
          <input type="file" id="file-input" accept=".csv,.xlsx,.xls" style="display:none" onchange="handleFileSelect(event)">
        </div>

        <!-- Sample Datasets -->
        <div class="glass-card no-hover">
          <h3 class="heading-4 mb-4">Or Try a Sample Dataset</h3>
          <p class="text-secondary mb-4" style="font-size:var(--text-sm)">Perfect for beginners! These classic datasets work out of the box.</p>
          <div id="sample-datasets" style="display:flex; flex-direction:column; gap:8px;">
            <div class="text-center p-4"><div class="loading-dots"><span></span><span></span><span></span></div></div>
          </div>
        </div>
      </div>

      <!-- Active Dataset Indicator -->
      <div id="active-dataset-bar" class="glass-card no-hover mb-6" style="display:none">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span style="font-size:24px">◉</span>
            <div>
              <div style="font-weight:600" id="active-dataset-name">—</div>
              <div class="text-secondary" style="font-size:var(--text-sm)" id="active-dataset-info">—</div>
            </div>
          </div>
          <div class="flex gap-2">
            <button class="btn btn-primary btn-sm" onclick="showEDA()">Auto-EDA</button>
            <button class="btn btn-outline btn-sm" onclick="showPreview()">Preview Data</button>
            <button class="btn btn-outline btn-sm" onclick="navigateTo('arena')">Train Models</button>
          </div>
        </div>
      </div>

      <!-- EDA Results -->
      <div id="eda-results"></div>
    </div>
  `;

  // Load sample datasets
  try {
    const data = await API.getSampleDatasets();
    const samplesEl = document.getElementById('sample-datasets');
    samplesEl.innerHTML = data.datasets.map(d => `
      <div class="sample-card" onclick="loadSample('${d.name}')">
        <div class="flex items-center justify-between">
          <div>
            <div class="sample-card-title">${d.title}</div>
            <div class="sample-card-desc">${d.description}</div>
          </div>
          <div class="flex gap-2">
            <span class="badge badge-${d.task === 'Classification' ? 'primary' : 'info'}">${d.task}</span>
            <span class="badge badge-${d.difficulty === 'Beginner' ? 'success' : 'warning'}">${d.difficulty}</span>
          </div>
        </div>
      </div>
    `).join('');
  } catch (e) {
    document.getElementById('sample-datasets').innerHTML = '<div class="text-muted">Failed to load samples</div>';
  }

  // If we already have a dataset loaded, show it
  if (AppState.currentDatasetId) {
    showActiveDataset();
  }
}

async function handleFileDrop(event) {
  event.preventDefault();
  event.currentTarget.classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file) await uploadFile(file);
}

async function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) await uploadFile(file);
}

async function uploadFile(file) {
  const dropzone = document.getElementById('dropzone');
  dropzone.innerHTML = `
    <div class="loading-dots"><span></span><span></span><span></span></div>
    <div class="dropzone-text mt-4">Uploading ${file.name}...</div>
  `;

  try {
    const result = await API.uploadDataset(file);
    AppState.currentDatasetId = result.id;
    AppState.currentDatasetName = result.name;
    AppState.datasets[result.id] = result;
    
    showToast(result.message, 'success');
    showActiveDataset();
    
    // Reset dropzone
    dropzone.innerHTML = `
      <div class="dropzone-icon">✅</div>
      <div class="dropzone-text">${file.name} uploaded!</div>
      <div class="dropzone-hint">Click to upload another file</div>
    `;
  } catch (e) {
    showToast(`Upload failed: ${e.message}`, 'error');
    dropzone.innerHTML = `
      <div class="dropzone-icon">📁</div>
      <div class="dropzone-text">Drag & drop your CSV file here</div>
      <div class="dropzone-hint">or click to browse • CSV, Excel supported</div>
    `;
  }
}

async function loadSample(name) {
  showToast(`Loading ${name} dataset...`, 'info');
  try {
    const result = await API.loadSampleDataset(name);
    AppState.currentDatasetId = result.id;
    AppState.currentDatasetName = result.name;
    AppState.datasets[result.id] = result;
    
    showToast(result.message, 'success');
    showActiveDataset();
  } catch (e) {
    showToast(`Failed to load: ${e.message}`, 'error');
  }
}

function showActiveDataset() {
  const bar = document.getElementById('active-dataset-bar');
  if (!bar) return;
  bar.style.display = 'block';
  
  const ds = AppState.datasets[AppState.currentDatasetId];
  if (ds) {
    document.getElementById('active-dataset-name').textContent = `${ds.name}`;
    const s = ds.stats;
    document.getElementById('active-dataset-info').textContent = 
      `${s.rows} rows × ${s.columns} columns • ${s.numeric_columns} numeric, ${s.categorical_columns} categorical • Quality: ${s.quality_score}/100`;
  }
}

async function showEDA() {
  if (!AppState.currentDatasetId) {
    showToast('Load a dataset first!', 'warning');
    return;
  }

  const edaEl = document.getElementById('eda-results');
  edaEl.innerHTML = `
    <div class="glass-card no-hover text-center p-6">
      <div class="loading-dots"><span></span><span></span><span></span></div>
      <p class="text-secondary mt-4">Running Auto-EDA analysis...</p>
    </div>
  `;

  try {
    const eda = await API.getEDA(AppState.currentDatasetId);
    
    edaEl.innerHTML = `
      <!-- Overview Section -->
      <div class="glass-card no-hover mb-6 animate-fade-in-up">
        <h3 class="heading-4 mb-6">Dataset Overview</h3>
        <div class="flex items-center gap-8 flex-wrap">
          ${qualityRingSVG(eda.overview.quality_score)}
          <div class="grid-4" style="flex:1">
            <div class="stat-card">
              <div class="stat-card-label">Rows</div>
              <div class="stat-card-value gradient">${eda.overview.rows.toLocaleString()}</div>
            </div>
            <div class="stat-card">
              <div class="stat-card-label">Columns</div>
              <div class="stat-card-value gradient">${eda.overview.columns}</div>
            </div>
            <div class="stat-card">
              <div class="stat-card-label">Missing</div>
              <div class="stat-card-value" style="color:${eda.overview.missing_percentage > 5 ? 'var(--warning)' : 'var(--success)'}">${eda.overview.missing_percentage}%</div>
            </div>
            <div class="stat-card">
              <div class="stat-card-label">Duplicates</div>
              <div class="stat-card-value" style="color:${eda.overview.duplicate_percentage > 5 ? 'var(--warning)' : 'var(--success)'}">${eda.overview.duplicate_rows}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Column Profiles -->
      <div class="glass-card no-hover mb-6 animate-fade-in-up">
        <h3 class="heading-4 mb-4">Column Profiles</h3>
        <p class="text-secondary mb-4" style="font-size:var(--text-sm)">Overview of every column in your dataset. Click column headers to sort.</p>
        <div class="data-table-wrapper" style="max-height:400px; overflow-y:auto">
          <table class="data-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Missing</th>
                <th>Unique</th>
                <th>Stats</th>
              </tr>
            </thead>
            <tbody>
              ${eda.columns.map(col => `
                <tr>
                  <td style="font-family:var(--font-sans); font-weight:600; color:var(--text-primary)">${col.name}</td>
                  <td><span class="badge badge-${col.is_numeric ? 'primary' : 'warning'}">${col.is_numeric ? 'Numeric' : 'Categorical'}</span></td>
                  <td>
                    ${col.missing > 0 
                      ? `<span style="color:var(--warning)">${col.missing} (${col.missing_pct}%)</span>` 
                      : '<span style="color:var(--success)">0 ✓</span>'}
                  </td>
                  <td>${col.unique} <span class="text-muted">(${col.unique_pct}%)</span></td>
                  <td style="font-family:var(--font-sans); font-size:var(--text-xs); color:var(--text-secondary)">
                    ${col.is_numeric 
                      ? `μ=${col.mean} σ=${col.std} [${col.min}, ${col.max}]`
                      : col.top_values ? `Top: ${Object.entries(col.top_values).slice(0,2).map(([k,v]) => `${k}(${v})`).join(', ')}` : '-'
                    }
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Distributions & Correlations -->
      <div class="grid-2 mb-6">
        <!-- Missing Values -->
        <div class="glass-card no-hover animate-fade-in-up">
          <h3 class="heading-4 mb-4">Missing Values</h3>
          ${eda.missing_matrix.counts.some(c => c > 0)
            ? generateBarChart(
                eda.missing_matrix.columns
                  .map((name, i) => ({ label: name, value: eda.missing_matrix.percentages[i] }))
                  .filter(d => d.value > 0)
                  .sort((a, b) => b.value - a.value),
                100,
                'var(--warning)'
              )
            : '<div class="text-center p-4" style="color:var(--success)">✅ No missing values! Your data is clean.</div>'
          }
        </div>

        <!-- Outliers -->
        <div class="glass-card no-hover animate-fade-in-up">
          <h3 class="heading-4 mb-4">⚠️ Outlier Detection</h3>
          ${Object.keys(eda.outliers).length > 0
            ? generateBarChart(
                Object.entries(eda.outliers)
                  .map(([name, info]) => ({ label: name, value: info.percentage }))
                  .filter(d => d.value > 0)
                  .sort((a, b) => b.value - a.value),
                null,
                'var(--error)'
              )
            : '<div class="text-center p-4 text-muted">No outliers detected via IQR method.</div>'
          }
        </div>
      </div>

      <!-- Correlation Matrix -->
      ${eda.correlations ? `
        <div class="glass-card no-hover mb-6 animate-fade-in-up">
          <h3 class="heading-4 mb-4">🔗 Correlation Heatmap</h3>
          <p class="text-secondary mb-4" style="font-size:var(--text-sm)">Shows linear relationships between numeric features. Values close to ±1 indicate strong correlation.</p>
          <div class="data-table-wrapper" style="max-height:400px; overflow:auto">
            <table class="data-table" style="font-size:11px">
              <thead>
                <tr>
                  <th style="position:sticky; left:0; background:var(--bg-surface); z-index:2"></th>
                  ${eda.correlations.columns.map(c => `<th style="writing-mode:vertical-lr; text-align:left; max-width:30px">${c.length > 15 ? c.slice(0,15)+'…' : c}</th>`).join('')}
                </tr>
              </thead>
              <tbody>
                ${eda.correlations.matrix.map((row, i) => `
                  <tr>
                    <td style="font-weight:600; font-family:var(--font-sans); position:sticky; left:0; background:var(--bg-card); z-index:1; white-space:nowrap">${eda.correlations.columns[i].length > 15 ? eda.correlations.columns[i].slice(0,15)+'…' : eda.correlations.columns[i]}</td>
                    ${row.map((val, j) => {
                      const abs = Math.abs(val);
                      const color = i === j ? 'rgba(99,102,241,0.3)' 
                        : val > 0 ? `rgba(99,102,241,${abs * 0.6})` 
                        : `rgba(239,68,68,${abs * 0.6})`;
                      return `<td style="background:${color}; text-align:center; color:${abs > 0.5 ? 'var(--text-primary)' : 'var(--text-muted)'}">${val.toFixed(2)}</td>`;
                    }).join('')}
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      ` : ''}

      <!-- Distributions -->
      ${Object.keys(eda.distributions).length > 0 ? `
        <div class="glass-card no-hover animate-fade-in-up">
          <h3 class="heading-4 mb-4">Feature Distributions</h3>
          <p class="text-secondary mb-4" style="font-size:var(--text-sm)">Histogram of values for each numeric column. Look for skewness and unusual patterns.</p>
          <div class="grid-3">
            ${Object.entries(eda.distributions).slice(0, 9).map(([col, dist]) => `
              <div style="background:var(--bg-surface); border-radius:var(--border-radius-md); padding:var(--space-4)">
                <div style="font-weight:600; font-size:var(--text-sm); margin-bottom:8px">${col}</div>
                <div style="display:flex; align-items:flex-end; gap:1px; height:60px;">
                  ${dist.histogram.counts.map((count, idx) => {
                    const maxCount = Math.max(...dist.histogram.counts);
                    const height = maxCount > 0 ? (count / maxCount * 100) : 0;
                    return `<div class="chart-bar" style="flex:1; height:${height}%; background:var(--primary); border-radius:2px 2px 0 0; min-width:2px; animation-delay:${idx * 30}ms" title="${count}"></div>`;
                  }).join('')}
                </div>
                <div class="flex justify-between mt-2" style="font-size:10px; color:var(--text-muted); font-family:var(--font-mono)">
                  <span>${dist.histogram.edges[0]}</span>
                  <span>${dist.histogram.edges[dist.histogram.edges.length - 1]}</span>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      ` : ''}
    `;
  } catch (e) {
    edaEl.innerHTML = `<div class="glass-card no-hover text-center" style="color:var(--error)">Failed to generate EDA: ${e.message}</div>`;
  }
}

async function showPreview() {
  if (!AppState.currentDatasetId) {
    showToast('Load a dataset first!', 'warning');
    return;
  }

  const edaEl = document.getElementById('eda-results');
  edaEl.innerHTML = `
    <div class="glass-card no-hover text-center p-6">
      <div class="loading-dots"><span></span><span></span><span></span></div>
      <p class="text-secondary mt-4">Loading data preview...</p>
    </div>
  `;

  try {
    const preview = await API.getPreview(AppState.currentDatasetId, 30);
    const colCount = preview.columns.length;

    edaEl.innerHTML = `
      <div class="glass-card no-hover animate-fade-in-up" style="padding:0; overflow:hidden">
        <div class="flex items-center justify-between" style="padding:var(--space-5) var(--space-6);">
          <div class="flex items-center gap-3">
            <h3 class="heading-4" style="margin:0">Data Preview</h3>
            <span class="badge badge-primary">${colCount} columns</span>
          </div>
          <span class="text-secondary" style="font-size:var(--text-xs)">Showing ${Math.min(30, preview.total_rows)} of ${preview.total_rows.toLocaleString()} rows</span>
        </div>
        <div class="preview-table-wrapper">
          <table class="preview-table">
            <thead>
              <tr>
                <th class="preview-row-num">#</th>
                ${preview.columns.map(c => `<th title="${c}">${c}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
              ${preview.data.map((row, i) => `
                <tr>
                  <td class="preview-row-num">${i + 1}</td>
                  ${row.map(val => {
                    const isNull = val === 'null' || val === null || val === undefined || val === '';
                    const strVal = isNull ? 'null' : String(val);
                    const display = strVal.length > 40 ? strVal.slice(0, 40) + '…' : strVal;
                    return `<td title="${strVal.replace(/"/g, '&quot;')}" class="${isNull ? 'null-val' : ''}">${display}</td>`;
                  }).join('')}
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        <div style="padding:var(--space-3) var(--space-6); border-top:var(--border-subtle); display:flex; align-items:center; justify-content:space-between;">
          <span class="text-muted" style="font-size:var(--text-xs)">Hover over cells to see full values · Scroll horizontally to see all columns</span>
          <span class="text-muted" style="font-size:var(--text-xs); font-family:var(--font-mono);">${preview.total_rows.toLocaleString()} × ${colCount}</span>
        </div>
      </div>
    `;
  } catch (e) {
    showToast(`Failed to load preview: ${e.message}`, 'error');
  }
}
