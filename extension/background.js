/**
 * Unified Phishing Detector - Background Service Worker (Manifest V3)
 * ===================================================================
 * Automatically analyzes visited URLs on navigation via the local FastAPI
 * backend server (Fast Path: <40ms ML prediction, no blocking threat-intel).
 * Manages tab action badges (Red '!' on Phishing, Green 'OK' on Safe)
 * and stores tab results in chrome.storage.local for instant popup access.
 */

const API_BASE_URL = 'http://127.0.0.1:8000';

/**
 * Validate whether a URL is an inspectable web resource (HTTP/HTTPS).
 */
function isInspectableUrl(url) {
  if (!url || typeof url !== 'string') return false;
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch (e) {
    return false;
  }
}

/**
 * Perform Fast-Path Phishing Analysis on a tab URL.
 */
async function analyzeTab(tabId, url) {
  if (!isInspectableUrl(url)) {
    // Clear badge for internal chrome://, extension pages, or about:blank
    try {
      await chrome.action.setBadgeText({ tabId, text: '' });
    } catch (e) {}
    return;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: url,
        check_threat_intel: false,
        generate_explanation: false,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const result = await response.json();
    const isPhishing = result.verdict === 'phishing';
    const confidencePct = Math.round((result.confidence || 0) * 100);

    // Update Action Badge based on verdict
    if (isPhishing) {
      await chrome.action.setBadgeText({ tabId, text: '!' });
      await chrome.action.setBadgeBackgroundColor({ tabId, color: '#D32F2F' }); // Vivid Crimson
      await chrome.action.setTitle({
        tabId,
        title: `Phishing Detected (${confidencePct}% confidence) - Click for details`,
      });
    } else {
      await chrome.action.setBadgeText({ tabId, text: 'OK' });
      await chrome.action.setBadgeBackgroundColor({ tabId, color: '#2E7D32' }); // Forest Green
      await chrome.action.setTitle({
        tabId,
        title: `Safe Website (${confidencePct}% confidence) - Unified Phishing Detector`,
      });
    }

    // Cache latest result in chrome.storage.local keyed by tab ID
    const storageKey = `tab_${tabId}`;
    const cachedEntry = {
      tabId: tabId,
      url: url,
      verdict: result.verdict,
      confidence: result.confidence,
      phishing_probability: result.phishing_probability,
      verdict_source: result.verdict_source,
      top_features: result.top_features || [],
      raw_scores: result.raw_scores || {},
      threat_intel: result.threat_intel || null,
      llm_explanation: result.llm_explanation || null,
      timestamp: Date.now(),
      server_offline: false,
    };

    await chrome.storage.local.set({ [storageKey]: cachedEntry });
  } catch (err) {
    clearTimeout(timeoutId);

    // Gracefully handle server offline without crashing or flooding console
    try {
      await chrome.action.setBadgeText({ tabId, text: '' });
      await chrome.action.setTitle({
        tabId,
        title: 'Unified Phishing Detector (Local Server Offline)',
      });

      const storageKey = `tab_${tabId}`;
      await chrome.storage.local.set({
        [storageKey]: {
          tabId: tabId,
          url: url,
          server_offline: true,
          error: err.name === 'AbortError' ? 'Connection timed out' : 'Server offline',
          timestamp: Date.now(),
        },
      });
    } catch (storageErr) {}
  }
}

// 1. Listen for tab URL navigation / updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab && tab.url) {
    analyzeTab(tabId, tab.url);
  }
});

// 2. Clean up storage when a tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  const storageKey = `tab_${tabId}`;
  chrome.storage.local.remove(storageKey).catch(() => {});
});

// 3. Handle messages from popup requesting immediate re-analysis
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYZE_TAB' && message.tabId && message.url) {
    analyzeTab(message.tabId, message.url).then(() => {
      sendResponse({ status: 'completed' });
    });
    return true; // Keep channel open for async response
  }
});
