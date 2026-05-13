/**
 * DataForge — Main App (SPA Router)
 */

const pages = {
  dashboard: renderDashboard,
  data: renderDataExplorer,
  pipeline: renderPipeline,
  arena: renderArena,
  experiments: renderExperiments,
  copilot: renderCopilot,
};

function navigateTo(page) {
  // Update active nav — handle both topnav-links and CTA button
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === page);
  });

  // Close mobile nav if open
  const links = document.getElementById('topnav-links');
  if (links) links.classList.remove('mobile-open');

  // Update state
  AppState.currentPage = page;
  window.location.hash = page;

  // Render page
  const container = document.getElementById('page-container');
  container.innerHTML = '<div class="text-center p-6"><div class="loading-dots"><span></span><span></span><span></span></div></div>';

  if (pages[page]) {
    // Small delay for transition feel
    setTimeout(() => {
      pages[page](container);
    }, 100);
  }
}

// Mobile nav toggle
function toggleMobileNav() {
  const links = document.getElementById('topnav-links');
  if (links) links.classList.toggle('mobile-open');
}

// Legacy compat for old sidebar toggle calls
function toggleMobileSidebar() {
  toggleMobileNav();
}

// Handle back/forward
window.addEventListener('hashchange', () => {
  const page = window.location.hash.replace('#', '') || 'dashboard';
  if (pages[page] && page !== AppState.currentPage) {
    navigateTo(page);
  }
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  const page = window.location.hash.replace('#', '') || 'dashboard';
  navigateTo(page);
});
