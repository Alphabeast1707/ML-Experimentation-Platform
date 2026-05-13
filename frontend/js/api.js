/**
 * DataForge — API Client
 * Handles all communication with the FastAPI backend
 */

const API = {
  baseUrl: '',

  async request(method, path, data = null) {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (data) options.body = JSON.stringify(data);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, options);
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`API Error [${method} ${path}]:`, error);
      throw error;
    }
  },

  // Data endpoints
  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${this.baseUrl}/api/data/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail);
    }
    return await response.json();
  },

  getSampleDatasets() { return this.request('GET', '/api/data/samples'); },
  loadSampleDataset(name) { return this.request('POST', `/api/data/samples/${name}`); },
  getEDA(datasetId) { return this.request('GET', `/api/data/${datasetId}/eda`); },
  getColumns(datasetId) { return this.request('GET', `/api/data/${datasetId}/columns`); },
  getPreview(datasetId, rows = 20) { return this.request('GET', `/api/data/${datasetId}/preview?rows=${rows}`); },

  // Model endpoints
  trainModel(config) { return this.request('POST', '/api/models/train', config); },
  compareModels(config) { return this.request('POST', '/api/models/compare', config); },

  // Experiment endpoints
  getExperiments() { return this.request('GET', '/api/experiments'); },
  getExperiment(id) { return this.request('GET', `/api/experiments/${id}`); },
  deleteExperiment(id) { return this.request('DELETE', `/api/experiments/${id}`); },

  // Pipeline endpoints
  executePipeline(config) { return this.request('POST', '/api/pipeline/execute', config); },

  // Copilot endpoints
  chatCopilot(message, context = {}) {
    return this.request('POST', '/api/copilot/chat', { message, context });
  },

  // Dashboard
  getDashboardStats() { return this.request('GET', '/api/dashboard/stats'); },
};
