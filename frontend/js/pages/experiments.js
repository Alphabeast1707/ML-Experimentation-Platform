/**
 * DataForge — Experiments Tracker Page
 */

async function renderExperiments(container) {
  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header-title">◇ Experiment Tracker</h1>
        <p class="page-header-desc">Every model you train is logged here. Compare runs, track progress, and find the best configuration.</p>
      </div>

      <div id="experiments-content">
        <div class="text-center p-6"><div class="loading-dots"><span></span><span></span><span></span></div></div>
      </div>
    </div>
  `;

  await loadExperiments();
}

async function loadExperiments() {
  const contentEl = document.getElementById('experiments-content');
  if (!contentEl) return;

  try {
    const data = await API.getExperiments();
    const experiments = data.experiments || [];
    AppState.experiments = experiments;

    // Update badge
    const badge = document.getElementById('exp-count');
    if (badge) {
      badge.textContent = experiments.length;
      badge.style.display = experiments.length > 0 ? 'inline' : 'none';
    }

    if (experiments.length === 0) {
      contentEl.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">◇</div>
          <div class="empty-state-title">No experiments yet</div>
          <div class="empty-state-description">Every model you train in the Model Arena gets tracked here automatically. Go start your first battle!</div>
          <button class="btn btn-gradient mt-4" onclick="navigateTo('arena')">Go to Model Arena</button>
        </div>
      `;
      return;
    }

    // Summary stats
    const bestClassAcc = Math.max(...experiments.filter(e => e.problem_type === 'classification').map(e => e.metrics?.accuracy || 0), 0);
    const bestRegR2 = Math.max(...experiments.filter(e => e.problem_type === 'regression').map(e => e.metrics?.r2 || 0), 0);
    const modelTypes = [...new Set(experiments.map(e => e.model_type))];

    contentEl.innerHTML = `
      <!-- Summary -->
      <div class="grid-4 mb-6 stagger-children">
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">Total Runs</div>
          <div class="stat-card-value gradient">${experiments.length}</div>
        </div>
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">Model Types</div>
          <div class="stat-card-value gradient">${modelTypes.length}</div>
        </div>
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">Best Accuracy</div>
          <div class="stat-card-value" style="color:var(--success)">${bestClassAcc > 0 ? (bestClassAcc * 100).toFixed(1) + '%' : '-'}</div>
        </div>
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">Best R²</div>
          <div class="stat-card-value" style="color:var(--success)">${bestRegR2 > 0 ? bestRegR2.toFixed(4) : '-'}</div>
        </div>
      </div>

      <!-- Experiments Table -->
      <div class="glass-card no-hover animate-fade-in-up">
        <div class="flex items-center justify-between mb-4">
          <h3 class="heading-4">All Experiments</h3>
          <span class="text-muted" style="font-size:var(--text-sm)">${experiments.length} runs total</span>
        </div>
        <div class="data-table-wrapper" style="max-height:600px; overflow-y:auto">
          <table class="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>Type</th>
                <th>Metrics</th>
                <th>CV Score</th>
                <th>Time</th>
                <th>When</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              ${experiments.map((exp, i) => {
                const m = exp.metrics || {};
                const isPrimaryBest = (exp.problem_type === 'classification' && m.accuracy === bestClassAcc) ||
                                     (exp.problem_type === 'regression' && m.r2 === bestRegR2);
                return `
                  <tr style="${isPrimaryBest ? 'background:rgba(16,185,129,0.05)' : ''}">
                    <td style="color:var(--text-muted)">${exp.id}</td>
                    <td style="font-family:var(--font-sans); font-weight:600; color:var(--text-primary)">
                      ${exp.model_type}
                      ${isPrimaryBest ? ' <span style="font-size:14px">★</span>' : ''}
                    </td>
                    <td style="font-family:var(--font-sans)">${exp.dataset_name}</td>
                    <td><span class="badge badge-${exp.problem_type === 'classification' ? 'primary' : 'info'}">${exp.problem_type}</span></td>
                    <td>
                      ${exp.problem_type === 'classification' 
                        ? `<span style="color:${metricColor(m.accuracy || 0)}; font-weight:600">${((m.accuracy || 0) * 100).toFixed(1)}%</span> acc`
                        : `<span style="color:${metricColor(m.r2 || 0)}; font-weight:600">${(m.r2 || 0).toFixed(4)}</span> R²`
                      }
                    </td>
                    <td>
                      ${m.cv_mean != null 
                        ? `${(m.cv_mean * 100).toFixed(1)}% <span class="text-muted">±${(m.cv_std * 100).toFixed(1)}%</span>`
                        : '<span class="text-muted">-</span>'
                      }
                    </td>
                    <td>${m.training_time ? m.training_time + 's' : '-'}</td>
                    <td style="font-family:var(--font-sans); color:var(--text-muted)">${formatDate(exp.created_at)}</td>
                    <td>
                      <button class="btn btn-ghost btn-sm" onclick="showExperimentDetail('${exp.id}')" title="View details">→</button>
                      <button class="btn btn-ghost btn-sm" onclick="deleteExperiment('${exp.id}')" title="Delete" style="color:var(--error)">🗑️</button>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  } catch (e) {
    contentEl.innerHTML = `<div class="glass-card no-hover" style="color:var(--error)">Failed to load experiments: ${e.message}</div>`;
  }
}

async function showExperimentDetail(id) {
  const exp = AppState.experiments.find(e => e.id === id);
  if (!exp) return;

  const m = exp.metrics || {};

  // Create modal
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

  overlay.innerHTML = `
    <div class="modal" style="max-width:700px">
      <div class="modal-header">
        <h3 class="modal-title">Experiment ${exp.id}</h3>
        <button class="btn btn-ghost btn-icon" onclick="this.closest('.modal-overlay').remove()">✕</button>
      </div>

      <div class="grid-2 mb-6">
        <div>
          <div class="form-label">Model</div>
          <div style="font-weight:600">${exp.model_type}</div>
        </div>
        <div>
          <div class="form-label">Dataset</div>
          <div style="font-weight:600">${exp.dataset_name}</div>
        </div>
        <div>
          <div class="form-label">Problem Type</div>
          <div><span class="badge badge-primary">${exp.problem_type}</span></div>
        </div>
        <div>
          <div class="form-label">Target Column</div>
          <div style="font-weight:600">${exp.target_column || '-'}</div>
        </div>
      </div>

      <h4 class="heading-4 mb-4">Metrics</h4>
      <div class="grid-3 mb-6">
        ${Object.entries(m).filter(([k]) => !['cv_mean', 'cv_std', 'training_time'].includes(k)).map(([key, val]) => `
          <div class="stat-card" style="padding:var(--space-3)">
            <div class="stat-card-label" style="font-size:11px; text-transform:uppercase">${key}</div>
            <div style="font-size:var(--text-lg); font-weight:700; font-family:var(--font-mono); color:${typeof val === 'number' && val <= 1 ? metricColor(val) : 'var(--text-primary)'}">
              ${typeof val === 'number' ? (val < 1 && val > 0 ? (val * 100).toFixed(2) + '%' : val.toFixed(4)) : val || '-'}
            </div>
          </div>
        `).join('')}
      </div>

      ${m.cv_mean != null ? `
        <div class="mb-6">
          <h4 class="heading-4 mb-2">🔄 Cross-Validation</h4>
          <p>Mean: <strong>${(m.cv_mean * 100).toFixed(2)}%</strong> ± ${(m.cv_std * 100).toFixed(2)}%</p>
        </div>
      ` : ''}

      ${exp.feature_columns && exp.feature_columns.length > 0 ? `
        <div class="mb-6">
          <h4 class="heading-4 mb-2">Features Used</h4>
          <div class="flex gap-2 flex-wrap">
            ${exp.feature_columns.map(f => `<span class="badge badge-primary">${f}</span>`).join('')}
          </div>
        </div>
      ` : ''}

      <div class="text-muted" style="font-size:var(--text-xs)">Created: ${formatDate(exp.created_at)}</div>
    </div>
  `;

  document.body.appendChild(overlay);
}

async function deleteExperiment(id) {
  if (!confirm('Delete this experiment?')) return;

  try {
    await API.deleteExperiment(id);
    showToast('Experiment deleted', 'success');
    await loadExperiments();
  } catch (e) {
    showToast(`Failed to delete: ${e.message}`, 'error');
  }
}
