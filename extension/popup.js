/**
 * Unified Phishing Detector - Popup Controller
 * ============================================
 * Coordinates popup UI rendering, tab state querying, fast ML verdict display,
 * and on-demand threat intelligence / LLM explanation calls.
 */

const API_BASE_URL = 'http://127.0.0.1:8000';

// DOM Elements
const serverStatusPill = document.getElementById('server-status-pill');
const serverStatusText = document.getElementById('server-status-text');
const offlineBanner = document.getElementById('offline-banner');
const btnRetryConnection = document.getElementById('btn-retry-connection');

const tabUrlEl = document.getElementById('tab-url');
const verdictCard = document.getElementById('verdict-card');
const verdictIcon = document.getElementById('verdict-icon');
const verdictBadge = document.getElementById('verdict-badge');
const confidenceText = document.getElementById('confidence-text');
const confidenceBarFill = document.getElementById('confidence-bar-fill');
const featuresChips = document.getElementById('features-chips');

const btnExplain = document.getElementById('btn-explain');
const btnExplainText = document.getElementById('btn-explain-text');
const explainLoading = document.getElementById('explain-loading');
const detailedResults = document.getElementById('detailed-results');
const actionPill = document.getElementById('action-pill');
const llmExplanationText = document.getElementById('llm-explanation-text');
const intelVtStatus = document.getElementById('intel-vt-status');
const intelGsbStatus = document.getElementById('intel-gsb-status');

// Local State
let activeTab = null;
let currentResult = null;

/**
 * Initialize popup upon opening.
 */
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await checkServerHealth();
  await loadActiveTabAnalysis();
});

function setupEventListeners() {
  btnExplain.addEventListener('click', handleGetDetailedExplanation);
  btnRetryConnection.addEventListener('click', async () => {
    await checkServerHealth();
    await loadActiveTabAnalysis(true);
  });
}

/**
 * Check if the FastAPI server is reachable.
 */
async function checkServerHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (res.ok) {
      setServerStatus(true);
      return true;
    } else {
      setServerStatus(false);
      return false;
    }
  } catch (e) {
    setServerStatus(false);
    return false;
  }
}

function setServerStatus(isOnline) {
  if (isOnline) {
    serverStatusPill.className = 'status-pill status-online';
    serverStatusText.textContent = 'Server Online';
    offlineBanner.classList.add('hidden');
    btnExplain.disabled = false;
  } else {
    serverStatusPill.className = 'status-pill status-offline';
    serverStatusText.textContent = 'Server Offline';
    offlineBanner.classList.remove('hidden');
    btnExplain.disabled = true;
  }
}

/**
 * Query current tab and load either cached results or perform fast ML query.
 */
async function loadActiveTabAnalysis(forceRefresh = false) {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tabs || tabs.length === 0) {
      tabUrlEl.textContent = 'No active tab detected';
      return;
    }

    activeTab = tabs[0];
    const url = activeTab.url || '';
    tabUrlEl.textContent = url;
    tabUrlEl.title = url;

    // Check URL scheme
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      renderNonWebPage(url);
      return;
    }

    const storageKey = `tab_${activeTab.id}`;
    const stored = await chrome.storage.local.get([storageKey]);
    const cachedData = stored[storageKey];

    if (!forceRefresh && cachedData && cachedData.url === url && !cachedData.server_offline) {
      // Use cached fast result
      currentResult = cachedData;
      renderVerdict(cachedData);

      // If detailed analysis was already cached, render it too
      if (cachedData.llm_explanation) {
        renderDetailedResults(cachedData);
      }
    } else {
      // Trigger fast path analysis
      await performFastAnalysis(activeTab.id, url);
    }
  } catch (err) {
    console.error('Error loading tab analysis:', err);
  }
}

/**
 * Run fast ML analysis (URL signals only).
 */
async function performFastAnalysis(tabId, url) {
  renderLoadingState('Analyzing URL structure & ML signals...');

  try {
    const res = await fetch(`${API_BASE_URL}/analyze-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: url,
        check_threat_intel: false,
        generate_explanation: false,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    currentResult = data;
    renderVerdict(data);
    setServerStatus(true);

    // Save to local storage
    const storageKey = `tab_${tabId}`;
    await chrome.storage.local.set({
      [storageKey]: {
        ...data,
        tabId: tabId,
        timestamp: Date.now(),
        server_offline: false,
      },
    });
  } catch (err) {
    setServerStatus(false);
    renderOfflineState();
  }
}

/**
 * Handle "Get Detailed Explanation" click (Slow Path: Threat Intel + LLM).
 */
async function handleGetDetailedExplanation() {
  if (!activeTab || !activeTab.url) return;

  btnExplain.disabled = true;
  explainLoading.classList.remove('hidden');
  detailedResults.classList.add('hidden');

  try {
    const res = await fetch(`${API_BASE_URL}/analyze-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: activeTab.url,
        check_threat_intel: true,
        generate_explanation: true,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const enrichedData = await res.json();
    currentResult = enrichedData;

    // Re-render primary verdict in case of Threat Intel escalation
    renderVerdict(enrichedData);

    // Render detailed explanation and intel
    renderDetailedResults(enrichedData);

    // Update storage with detailed data
    const storageKey = `tab_${activeTab.id}`;
    await chrome.storage.local.set({
      [storageKey]: {
        ...enrichedData,
        tabId: activeTab.id,
        timestamp: Date.now(),
        server_offline: false,
      },
    });
  } catch (err) {
    console.error('Detailed explanation error:', err);
    alert('Failed to retrieve detailed explanation. Check backend server status.');
  } finally {
    explainLoading.classList.add('hidden');
    btnExplain.disabled = false;
  }
}

/**
 * Render verdict card.
 */
function renderVerdict(data) {
  const isPhishing = data.verdict === 'phishing';
  const confidencePct = Math.round(data.confidence * 100);

  // Update card styling
  verdictCard.className = `card verdict-card ${isPhishing ? 'verdict-phishing' : 'verdict-safe'}`;
  verdictIcon.textContent = isPhishing ? '⚠️' : '🛡️';
  verdictBadge.textContent = isPhishing ? 'PHISHING DETECTED' : 'SAFE WEBSITE';

  confidenceText.textContent = `${confidencePct}% Confidence (${data.verdict_source === 'threat_intel_escalation' ? 'Threat Intel Escalation' : 'AI Stacking Ensemble'})`;
  confidenceBarFill.style.width = `${confidencePct}%`;

  // Render top feature chips
  renderFeatureChips(data.top_features || [], isPhishing);
}

/**
 * Render feature chips.
 */
function renderFeatureChips(features, isPhishing) {
  featuresChips.innerHTML = '';
  if (!features || features.length === 0) {
    const chip = document.createElement('span');
    chip.className = 'chip chip-neutral';
    chip.textContent = 'Standard profile';
    featuresChips.appendChild(chip);
    return;
  }

  features.forEach((feat) => {
    const chip = document.createElement('span');
    chip.className = `chip ${isPhishing ? 'chip-phish' : 'chip-safe'}`;
    chip.textContent = feat;
    featuresChips.appendChild(chip);
  });
}

/**
 * Render Detailed LLM Explanation & Threat Intelligence.
 */
function renderDetailedResults(data) {
  detailedResults.classList.remove('hidden');

  // 1. Recommended Action Pill
  const action = (data.llm_explanation?.recommended_action || (data.verdict === 'phishing' ? 'block' : 'safe')).toLowerCase();
  actionPill.textContent = action.toUpperCase();

  if (action === 'block') {
    actionPill.className = 'action-pill action-block';
  } else if (action === 'review') {
    actionPill.className = 'action-pill action-review';
  } else {
    actionPill.className = 'action-pill action-safe';
  }

  // 2. LLM Explanation Text
  if (data.llm_explanation?.explanation) {
    llmExplanationText.textContent = data.llm_explanation.explanation;
  } else {
    llmExplanationText.textContent = 'No natural language explanation generated for this URL.';
  }

  // 3. Threat Intel Sources
  const threatIntel = data.threat_intel;
  if (threatIntel && threatIntel.sources) {
    const vt = threatIntel.sources.virustotal;
    const gsb = threatIntel.sources.google_safe_browsing;

    if (vt) {
      if (vt.flagged) {
        intelVtStatus.textContent = `Flagged (${vt.malicious_count || 0}/${vt.total_engines || 0} engines)`;
        intelVtStatus.className = 'intel-val flagged';
      } else {
        intelVtStatus.textContent = `Clean (0/${vt.total_engines || 0} engines)`;
        intelVtStatus.className = 'intel-val clean';
      }
    } else {
      intelVtStatus.textContent = 'Not Checked';
      intelVtStatus.className = 'intel-val';
    }

    if (gsb) {
      if (gsb.flagged) {
        intelGsbStatus.textContent = 'Flagged (Malicious)';
        intelGsbStatus.className = 'intel-val flagged';
      } else {
        intelGsbStatus.textContent = 'Clean';
        intelGsbStatus.className = 'intel-val clean';
      }
    } else {
      intelGsbStatus.textContent = 'Not Checked';
      intelGsbStatus.className = 'intel-val';
    }
  } else {
    intelVtStatus.textContent = 'No Intel Data';
    intelGsbStatus.textContent = 'No Intel Data';
  }

  btnExplainText.textContent = 'Refresh Detailed Explanation';
}

function renderLoadingState(msg) {
  verdictCard.className = 'card verdict-card verdict-neutral';
  verdictIcon.textContent = '⏳';
  verdictBadge.textContent = 'ANALYZING...';
  confidenceText.textContent = msg;
  confidenceBarFill.style.width = '30%';
}

function renderOfflineState() {
  verdictCard.className = 'card verdict-card verdict-neutral';
  verdictIcon.textContent = '🔌';
  verdictBadge.textContent = 'SERVER OFFLINE';
  confidenceText.textContent = 'Backend server unavailable';
  confidenceBarFill.style.width = '0%';
  featuresChips.innerHTML = '<span class="chip chip-neutral">Server not running</span>';
}

function renderNonWebPage(url) {
  verdictCard.className = 'card verdict-card verdict-neutral';
  verdictIcon.textContent = 'ℹ️';
  verdictBadge.textContent = 'INTERNAL PAGE';
  confidenceText.textContent = 'Internal browser or non-HTTP pages are not inspected';
  confidenceBarFill.style.width = '0%';
  featuresChips.innerHTML = '<span class="chip chip-neutral">Internal URL</span>';
  btnExplain.disabled = true;
}
