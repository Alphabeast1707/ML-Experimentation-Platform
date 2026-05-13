/**
 * DataForge — Utility Functions
 */

// Global app state
const AppState = {
  currentPage: 'dashboard',
  currentDatasetId: null,
  currentDatasetName: null,
  datasets: {},
  experiments: [],
};

// Toast notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
  
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type]}</span>
    <span class="toast-message">${message}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(toast);
  
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(30px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Simple markdown renderer
function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Headers
    .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold & italic
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Horizontal rule
    .replace(/^---$/gm, '<hr>')
    // Tables
    .replace(/^\|(.+)\|$/gm, (match) => {
      const cells = match.split('|').filter(c => c.trim());
      if (cells.every(c => /^[\s-:]+$/.test(c))) return '<!--table-sep-->';
      return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
    })
    // Unordered lists  
    .replace(/^- \[x\] (.+)$/gm, '<li class="checked">✅ $1</li>')
    .replace(/^- \[ \] (.+)$/gm, '<li class="unchecked">☐ $1</li>')
    .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
    // Ordered lists
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // Line breaks
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
  
  // Wrap consecutive <li> in <ul>
  html = html.replace(/((?:<li[^>]*>.*?<\/li><br>?)+)/g, '<ul>$1</ul>');
  // Wrap consecutive <tr> in <table>
  html = html.replace(/((?:<tr>.*?<\/tr>(?:<br>)?)+)/g, '<table>$1</table>');
  html = html.replace(/<!--table-sep--><br>?/g, '');
  // Clean up extra <br> inside lists/tables
  html = html.replace(/<\/li><br>/g, '</li>');
  html = html.replace(/<\/tr><br>/g, '</tr>');
  
  return html;
}

// Mobile nav toggle
function toggleMobileSidebar() {
  const links = document.getElementById('topnav-links');
  if (links) links.classList.toggle('mobile-open');
}

// Generate quality ring SVG
function qualityRingSVG(score) {
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  
  let color = '#10b981'; // green
  if (score < 60) color = '#ef4444'; // red
  else if (score < 80) color = '#f59e0b'; // yellow
  
  return `
    <div class="quality-ring">
      <svg width="120" height="120">
        <circle cx="60" cy="60" r="${radius}" stroke="rgba(255,255,255,0.06)" stroke-width="8" fill="none"/>
        <circle cx="60" cy="60" r="${radius}" stroke="${color}" stroke-width="8" fill="none"
                stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"
                stroke-linecap="round" style="transition: stroke-dashoffset 1s ease"/>
      </svg>
      <div style="position:absolute; text-align:center">
        <div class="quality-ring-value" style="color:${color}">${score}</div>
        <div class="quality-ring-label">Quality</div>
      </div>
    </div>
  `;
}

// Animated counter
function animateCounter(element, target, duration = 800) {
  const start = 0;
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease out cubic
    const current = start + (target - start) * eased;
    
    if (Number.isInteger(target)) {
      element.textContent = Math.round(current);
    } else {
      element.textContent = current.toFixed(4);
    }
    
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Format date
function formatDate(isoString) {
  const d = new Date(isoString);
  const now = new Date();
  const diffMs = now - d;
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// Generate a simple bar chart with CSS
function generateBarChart(data, maxValue, color = 'var(--primary)') {
  if (!data || data.length === 0) return '<div class="text-muted text-center p-4">No data</div>';
  
  const max = maxValue || Math.max(...data.map(d => d.value));
  
  return `
    <div style="display:flex; flex-direction:column; gap:8px;">
      ${data.slice(0, 10).map((d, i) => `
        <div class="metric-bar" style="animation-delay:${i * 50}ms">
          <span class="metric-bar-label" title="${d.label}">${d.label.length > 12 ? d.label.slice(0, 12) + '…' : d.label}</span>
          <div class="metric-bar-track">
            <div class="metric-bar-fill chart-bar" style="width:${max > 0 ? (d.value / max * 100) : 0}%; background:${color}; animation-delay:${i * 80}ms"></div>
          </div>
          <span class="metric-bar-value">${typeof d.value === 'number' ? (d.value < 1 ? d.value.toFixed(4) : d.value.toFixed(2)) : d.value}</span>
        </div>
      `).join('')}
    </div>
  `;
}

// Generate confusion matrix HTML
function generateConfusionMatrix(matrix) {
  if (!matrix || matrix.length === 0) return '';
  const n = matrix.length;
  
  return `
    <div class="confusion-matrix" style="grid-template-columns: repeat(${n}, 1fr);">
      ${matrix.map((row, i) =>
        row.map((val, j) => `
          <div class="confusion-cell ${i === j ? 'diagonal' : 'off-diagonal'}">
            ${val}
          </div>
        `).join('')
      ).join('')}
    </div>
  `;
}

// Metric color based on value
function metricColor(value) {
  if (value >= 0.9) return 'var(--success)';
  if (value >= 0.7) return 'var(--primary)';
  if (value >= 0.5) return 'var(--warning)';
  return 'var(--error)';
}

// Debounce utility
function debounce(func, wait) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}
