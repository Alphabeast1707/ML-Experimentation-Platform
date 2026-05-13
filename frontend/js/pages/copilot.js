/**
 * DataForge — AI Copilot Page
 */

let chatHistory = [];

async function renderCopilot(container) {
  container.innerHTML = `
    <div class="page" style="height: calc(100vh - 64px); display:flex; flex-direction:column">
      <div class="page-header" style="flex-shrink:0">
        <h1 class="page-header-title">✦ AI Copilot</h1>
        <p class="page-header-desc">Your intelligent ML assistant. Ask anything about machine learning, your data, or how to improve your models.</p>
      </div>

      <div class="glass-card no-hover" style="flex:1; display:flex; flex-direction:column; overflow:hidden; padding:0">
        <!-- Quick Suggestions -->
        <div id="copilot-suggestions" style="padding:var(--space-4); border-bottom:var(--border-subtle); flex-shrink:0; overflow-x:auto; white-space:nowrap;">
          <span class="text-muted" style="font-size:var(--text-xs); margin-right:8px">Try asking:</span>
          ${[
            'How to get started?',
            'Which model should I use?',
            'My model is overfitting',
            'Feature engineering tips',
            'Explain metrics',
            'Handle missing data',
            'Handle imbalanced data'
          ].map(q => `
            <button class="btn btn-outline btn-sm" onclick="askCopilot('${q}')" style="display:inline-flex; margin-right:4px; white-space:nowrap">${q}</button>
          `).join('')}
        </div>

        <!-- Chat Messages -->
        <div class="chat-messages" id="chat-messages">
          ${chatHistory.length === 0 ? `
            <div class="chat-message bot animate-fade-in-up">
              <div class="chat-avatar">✦</div>
              <div class="chat-bubble">
                <h1>Hi! I'm your ML Copilot</h1>
                <p>I'm here to help you build better machine learning models. I can:</p>
                <ul>
                  <li><strong>Recommend models</strong> based on your data</li>
                  <li><strong>Diagnose problems</strong> like overfitting or low accuracy</li>
                  <li><strong>Explain concepts</strong> in simple terms</li>
                  <li><strong>Guide you step by step</strong> through the ML workflow</li>
                </ul>
                <p>Just type a question below, or click one of the quick suggestions above!</p>
              </div>
            </div>
          ` : chatHistory.map(msg => renderChatMessage(msg)).join('')}
        </div>

        <!-- Input Bar -->
        <div class="chat-input-bar">
          <input type="text" class="form-input" id="copilot-input" 
                 placeholder="Ask me anything about ML..." 
                 onkeydown="if(event.key==='Enter') sendCopilotMessage()"
                 autocomplete="off">
          <button class="btn btn-primary" onclick="sendCopilotMessage()" id="copilot-send-btn">
            Send ➤
          </button>
        </div>
      </div>
    </div>
  `;

  // Focus input
  setTimeout(() => document.getElementById('copilot-input')?.focus(), 300);
}

function renderChatMessage(msg) {
  if (msg.role === 'user') {
    return `
      <div class="chat-message user animate-fade-in-up">
        <div class="chat-avatar">◈</div>
        <div class="chat-bubble">${escapeHtml(msg.content)}</div>
      </div>
    `;
  } else {
    return `
      <div class="chat-message bot animate-fade-in-up">
        <div class="chat-avatar">✦</div>
        <div class="chat-bubble">${renderMarkdown(msg.content)}</div>
      </div>
    `;
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function askCopilot(question) {
  const input = document.getElementById('copilot-input');
  if (input) {
    input.value = question;
    sendCopilotMessage();
  }
}

async function sendCopilotMessage() {
  const input = document.getElementById('copilot-input');
  const message = input?.value?.trim();
  if (!message) return;

  // Add user message
  chatHistory.push({ role: 'user', content: message });
  input.value = '';

  const messagesEl = document.getElementById('chat-messages');
  messagesEl.innerHTML += renderChatMessage({ role: 'user', content: message });

  // Add loading indicator
  const loadingId = 'loading-' + Date.now();
  messagesEl.innerHTML += `
    <div class="chat-message bot animate-fade-in-up" id="${loadingId}">
      <div class="chat-avatar">🤖</div>
      <div class="chat-bubble">
        <div class="loading-dots"><span></span><span></span><span></span></div>
        <span class="text-muted" style="font-size:var(--text-xs); margin-left:8px">Thinking...</span>
      </div>
    </div>
  `;
  messagesEl.scrollTop = messagesEl.scrollHeight;

  // Disable send
  const btn = document.getElementById('copilot-send-btn');
  if (btn) btn.disabled = true;

  try {
    const result = await API.chatCopilot(message, {
      dataset_id: AppState.currentDatasetId || null
    });

    // Remove loading
    document.getElementById(loadingId)?.remove();

    // Add bot response
    const botMessage = { role: 'bot', content: result.response };
    chatHistory.push(botMessage);
    messagesEl.innerHTML += renderChatMessage(botMessage);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (e) {
    document.getElementById(loadingId)?.remove();
    messagesEl.innerHTML += `
      <div class="chat-message bot animate-fade-in-up">
        <div class="chat-avatar">✦</div>
        <div class="chat-bubble" style="border-color:var(--error)">
          Sorry, I encountered an error: ${e.message}. Please try again!
        </div>
      </div>
    `;
  } finally {
    if (btn) btn.disabled = false;
    input?.focus();
  }
}
