/**
 * DataForge — Dashboard Page
 */

async function renderDashboard(container) {
  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header-title">Welcome to <span class="text-gradient">DataForge</span></h1>
        <p class="page-header-desc">Your ML command center. Upload data, train models, compare results — all in one place.</p>
      </div>

      <!-- Stats -->
      <div class="grid-4 stagger-children mb-8" id="dashboard-stats">
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">◫ Datasets</div>
          <div class="stat-card-value gradient" id="stat-datasets">-</div>
          <div class="stat-card-change">Loaded in this session</div>
        </div>
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">◇ Experiments</div>
          <div class="stat-card-value gradient" id="stat-experiments">-</div>
          <div class="stat-card-change">Models trained & tracked</div>
        </div>
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">△ Models</div>
          <div class="stat-card-value gradient" id="stat-models">-</div>
          <div class="stat-card-change">Ready for comparison</div>
        </div>
        <div class="stat-card animate-fade-in-up">
          <div class="stat-card-label">◎ Best Accuracy</div>
          <div class="stat-card-value gradient" id="stat-accuracy">-</div>
          <div class="stat-card-change">Across all experiments</div>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="glass-card no-hover mb-8">
        <h3 class="heading-4 mb-4">Quick Actions</h3>
        <p class="text-secondary mb-6" style="font-size:var(--text-sm)">New here? Follow these steps to run your first ML experiment in under 2 minutes!</p>
        <div class="steps mb-6" id="onboarding-steps">
          <div class="step ${!AppState.currentDatasetId ? 'active' : 'completed'}" id="step-1">
            <div class="step-number">1</div>
            <span class="step-label">Load Data</span>
          </div>
          <div class="step-connector"></div>
          <div class="step" id="step-2">
            <div class="step-number">2</div>
            <span class="step-label">Explore</span>
          </div>
          <div class="step-connector"></div>
          <div class="step" id="step-3">
            <div class="step-number">3</div>
            <span class="step-label">Train Models</span>
          </div>
          <div class="step-connector"></div>
          <div class="step" id="step-4">
            <div class="step-number">4</div>
            <span class="step-label">Compare</span>
          </div>
        </div>
        <div class="flex gap-3 flex-wrap">
          <button class="btn btn-gradient btn-lg" onclick="navigateTo('data')">
            Upload Dataset
          </button>
          <button class="btn btn-outline btn-lg" onclick="navigateTo('arena')">
            Model Arena
          </button>
          <button class="btn btn-outline btn-lg" onclick="navigateTo('copilot')">
            Ask AI Copilot
          </button>
        </div>
      </div>

      <!-- Recent Experiments -->
      <div class="glass-card no-hover">
        <div class="flex items-center justify-between mb-4">
          <h3 class="heading-4">Recent Experiments</h3>
          <button class="btn btn-ghost btn-sm" onclick="navigateTo('experiments')">View All →</button>
        </div>
        <div id="recent-experiments">
          <div class="empty-state" style="padding:var(--space-8)">
            <div class="empty-state-icon">◇</div>
            <div class="empty-state-title">No experiments yet</div>
            <div class="empty-state-description">Train your first model in the Model Arena to see results here!</div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Fetch stats
  try {
    const stats = await API.getDashboardStats();
    const ds = document.getElementById('stat-datasets');
    const ex = document.getElementById('stat-experiments');
    const mo = document.getElementById('stat-models');
    const ac = document.getElementById('stat-accuracy');
    
    if (ds) animateCounter(ds, stats.total_datasets);
    if (ex) animateCounter(ex, stats.total_experiments);
    if (mo) animateCounter(mo, stats.total_models);
    if (ac) {
      ac.textContent = stats.best_accuracy > 0 ? (stats.best_accuracy * 100).toFixed(1) + '%' : '-';
    }

    // Update onboarding steps
    if (stats.total_datasets > 0) {
      document.getElementById('step-1')?.classList.replace('active', 'completed');
      document.getElementById('step-2')?.classList.add('active');
    }
    if (stats.total_experiments > 0) {
      document.getElementById('step-2')?.classList.replace('active', 'completed');
      document.getElementById('step-3')?.classList.replace('active', 'completed');
      document.getElementById('step-4')?.classList.add('active');
    }

    // Update experiment badge
    const badge = document.getElementById('exp-count');
    if (badge && stats.total_experiments > 0) {
      badge.textContent = stats.total_experiments;
      badge.style.display = 'inline';
    }

    // Recent experiments
    if (stats.recent_experiments && stats.recent_experiments.length > 0) {
      const recentEl = document.getElementById('recent-experiments');
      recentEl.innerHTML = `
        <div class="data-table-wrapper">
          <table class="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Dataset</th>
                <th>Type</th>
                <th>Primary Metric</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              ${stats.recent_experiments.map(exp => {
                const metric = exp.problem_type === 'classification' 
                  ? `${((exp.metrics?.accuracy || 0) * 100).toFixed(1)}% accuracy`
                  : `${(exp.metrics?.r2 || 0).toFixed(4)} R²`;
                return `
                  <tr>
                    <td style="color:var(--text-primary); font-family:var(--font-sans); font-weight:500">${exp.model_type}</td>
                    <td style="font-family:var(--font-sans)">${exp.dataset_name}</td>
                    <td><span class="badge badge-${exp.problem_type === 'classification' ? 'primary' : 'info'}">${exp.problem_type}</span></td>
                    <td style="color:${metricColor(exp.metrics?.accuracy || exp.metrics?.r2 || 0)}">${metric}</td>
                    <td style="font-family:var(--font-sans); color:var(--text-muted)">${formatDate(exp.created_at)}</td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      `;
    }
  } catch (e) {
    console.error('Failed to load dashboard stats:', e);
  }
}
