/* ============================================================
   BHAIRAV AI — FRONTEND LOGIC (app.js)
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const submitBtn = document.getElementById("submit-btn");
  const topKInput = document.getElementById("setting-top-k");
  const citationsCheckbox = document.getElementById("setting-citations");
  const queryHistoryContainer = document.getElementById("query-history");
  const systemStatusIndicator = document.getElementById("system-status-indicator");
  
  // States
  const welcomeState = document.getElementById("welcome-state");
  const loadingState = document.getElementById("loading-state");
  const errorState = document.getElementById("error-state");
  const resultsState = document.getElementById("results-state");
  const errorMessage = document.getElementById("error-message");
  const errorResetBtn = document.getElementById("error-reset-btn");
  
  // Results Elements
  const resultIntent = document.getElementById("result-intent");
  const resultConfidence = document.getElementById("result-confidence");
  const resultSources = document.getElementById("result-sources");
  const resultTime = document.getElementById("result-time");
  const answerContainer = document.getElementById("answer-container");
  const sourcesContainer = document.getElementById("sources-container");
  const citationCount = document.getElementById("citation-count");
  
  // Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  // System Stats
  const statVectors = document.getElementById("stat-vectors");
  const statBm25 = document.getElementById("stat-bm25");
  const statEmbModel = document.getElementById("stat-emb-model");
  const statLlmModel = document.getElementById("stat-llm-model");

  // Local Storage for Query History
  let searchHistory = JSON.parse(localStorage.getItem("bhairav_history")) || [];

  // Initialize
  initApp();

  function initApp() {
    fetchSystemHealth();
    renderHistory();
    setupEventListeners();
  }

  // Setup UI Listeners
  function setupEventListeners() {
    // Form Submit
    queryForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const query = queryInput.value.trim();
      if (query) {
        executeSearch(query);
      }
    });

    // Reset Error View
    errorResetBtn.addEventListener("click", () => {
      showState("welcome");
    });

    // Tab Switching
    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        
        tabButtons.forEach(b => b.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));
        
        btn.classList.add("active");
        document.getElementById(targetTab).classList.add("active");
      });
    });

    // Suggestion Tag Clicks
    document.querySelectorAll(".suggestion-tag").forEach(tag => {
      tag.addEventListener("click", () => {
        queryInput.value = tag.textContent;
        queryInput.focus();
        executeSearch(tag.textContent);
      });
    });
  }

  // Fetch API Health Stats
  async function fetchSystemHealth() {
    updateStatus("connecting");
    try {
      const response = await fetch("/health");
      if (!response.ok) throw new Error("Backend unavailable");
      
      const data = await response.json();
      
      statVectors.textContent = data.faiss_vectors.toLocaleString();
      statBm25.textContent = data.bm25_corpus_size.toLocaleString();
      statEmbModel.textContent = cleanModelName(data.embedding_model);
      statLlmModel.textContent = cleanModelName(data.llm_model);
      
      updateStatus("ready");
    } catch (err) {
      console.error("Health check error:", err);
      updateStatus("offline");
    }
  }

  function cleanModelName(name) {
    if (!name) return "Unknown";
    // Shorten model paths/names for display
    if (name.includes("/")) {
      return name.split("/").pop();
    }
    return name;
  }

  // Set system status banner
  function updateStatus(status) {
    const dot = systemStatusIndicator.querySelector(".status-dot");
    const text = systemStatusIndicator.querySelector(".status-text");
    
    dot.className = "status-dot";
    
    if (status === "ready") {
      dot.classList.add("online");
      text.textContent = "SHASTRA INDEX ONLINE";
    } else if (status === "connecting") {
      dot.classList.add("loading");
      text.textContent = "CONNECTING TO SHASTRA...";
    } else {
      dot.classList.add("offline");
      text.textContent = "SHASTRA OFFLINE";
    }
  }

  // Toggle visible workspace state panel
  function showState(state) {
    welcomeState.classList.remove("active");
    loadingState.classList.remove("active");
    errorState.classList.remove("active");
    resultsState.classList.remove("active");

    if (state === "welcome") welcomeState.classList.add("active");
    else if (state === "loading") loadingState.classList.add("active");
    else if (state === "error") errorState.classList.add("active");
    else if (state === "results") resultsState.classList.add("active");
  }

  // Stepper animation timeouts
  let stepperTimeouts = [];
  function startLoadingAnimation() {
    // Clear any active timeouts
    stepperTimeouts.forEach(t => clearTimeout(t));
    stepperTimeouts = [];

    const steps = [
      { id: "step-normalizer", activeText: "Parsing query through Vidyut and resolving Sandhis..." },
      { id: "step-retrieval", activeText: "Querying FAISS vector index and BM25 hybrid pipeline..." },
      { id: "step-reranker", activeText: "Reranking candidates using Cross-Encoder model..." },
      { id: "step-generator", activeText: "Generating evergreen scholarly answer using Llama 3.3..." }
    ];

    // Reset all steps to default
    steps.forEach((step, idx) => {
      const el = document.getElementById(step.id);
      if (el) {
        el.className = "step" + (idx === 0 ? " active" : "");
      }
    });

    const statusText = document.getElementById("loading-status-text");
    const substatusText = document.querySelector(".loading-substatus");

    if (statusText) statusText.textContent = steps[0].activeText;
    if (substatusText) substatusText.textContent = "Processing step 1 of 4";

    // Schedule transitions
    const transitions = [
      { delay: 800, nextIdx: 1 },
      { delay: 2200, nextIdx: 2 },
      { delay: 3500, nextIdx: 3 }
    ];

    transitions.forEach(trans => {
      const timeout = setTimeout(() => {
        // Complete all steps before nextIdx
        for (let i = 0; i < trans.nextIdx; i++) {
          const el = document.getElementById(steps[i].id);
          if (el) el.className = "step completed";
        }
        // Activate nextIdx
        const nextEl = document.getElementById(steps[trans.nextIdx].id);
        if (nextEl) nextEl.className = "step active";
        if (statusText) statusText.textContent = steps[trans.nextIdx].activeText;
        if (substatusText) substatusText.textContent = `Processing step ${trans.nextIdx + 1} of 4`;
      }, trans.delay);
      stepperTimeouts.push(timeout);
    });
  }

  function stopLoadingAnimation() {
    stepperTimeouts.forEach(t => clearTimeout(t));
    stepperTimeouts = [];
  }

  // Execute RAG Pipeline Query
  async function executeSearch(query) {
    showState("loading");
    startLoadingAnimation();
    
    // Save to history list
    addToHistory(query);

    const startTime = performance.now();
    const payload = {
      query: query,
      top_k: parseInt(topKInput.value) || 10,
      include_citations: citationsCheckbox.checked
    };

    try {
      const response = await fetch("/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server failed to process retrieval request");
      }

      const data = await response.json();
      const endTime = performance.now();
      const localLatency = ((endTime - startTime) / 1000).toFixed(2);

      // Render Results
      renderResults(data, localLatency);
      stopLoadingAnimation();
      showState("results");

    } catch (err) {
      console.error("Search API failed:", err);
      stopLoadingAnimation();
      errorMessage.textContent = err.message || "Failed to retrieve or generate a response. Please check server logs.";
      showState("error");
    }
  }

  // Format and Render the response layout
  function renderResults(data, localLatency) {
    // 1. Update Stamp Metadata
    resultIntent.textContent = data.intent || "FACTUAL";
    
    const confidence = (data.confidence_band || "medium").toUpperCase();
    resultConfidence.textContent = confidence;
    resultConfidence.className = "value";
    if (confidence === "HIGH") {
      resultConfidence.style.color = "#2E7D32";
    } else if (confidence === "LOW") {
      resultConfidence.style.color = "#D32F2F";
    } else {
      resultConfidence.style.color = "var(--color-charcoal)";
    }

    resultSources.textContent = data.sources_used.length > 0 ? data.sources_used.join(", ") : "None";
    resultTime.textContent = `${localLatency}s`;

    // 2. Render synthesized markdown text
    answerContainer.innerHTML = parseMarkdown(data.answer);

    // 3. Clear and Render Retrieved Cards
    sourcesContainer.innerHTML = "";
    const citationsMap = new Map();
    
    // Build quick lookup for citations by chunk-like ID
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach(c => {
        citationsMap.set(c.id, c);
      });
    }

    const chunks = data.retrieved_chunks || [];
    citationCount.textContent = chunks.length;

    if (chunks.length === 0) {
      sourcesContainer.innerHTML = `<div class="empty-history" style="text-align:center; padding: 40px;">No original scriptural chunks were retrieved for this query.</div>`;
      return;
    }

    chunks.forEach((chunk) => {
      // Find corresponding citation text if available
      const citation = citationsMap.get(chunk.id);
      const textDevanagari = citation ? citation.text : "";
      
      const card = document.createElement("div");
      card.className = "source-card";
      
      // Formatting location label with dot delimiters
      let locLabel = chunk.source;
      let coords = [];
      if (chunk.book) coords.push(chunk.book);
      if (chunk.chapter) coords.push(`Ch.${chunk.chapter}`);
      if (chunk.verse) coords.push(`v.${chunk.verse}`);
      if (coords.length > 0) {
        locLabel += ` &middot; ` + coords.join(" &middot; ");
      }

      card.innerHTML = `
        <div class="source-card-header">
          <div class="source-provenance-row">
            <div class="source-provenance-tags">
              <span class="source-chunk-tag">CHUNK ${chunk.id}</span>
              <span class="source-score-tag">SCORE ${chunk.score.toFixed(4)}</span>
            </div>
            <span class="source-tier">TIER ${chunk.tier}</span>
          </div>
          <div class="source-location">${locLabel}</div>
        </div>
        <div class="source-card-body">
          ${textDevanagari ? `<div class="source-sanskrit-verse">&ldquo;${textDevanagari}&rdquo;</div>` : ""}
          <div class="source-hindi-summary">${chunk.hindi_summary}</div>
        </div>
        <div class="source-card-footer">
          <span class="source-meta-item">CRITICAL EDITION APPARATUS</span>
          <span class="source-meta-item">INDEX PATH: <span>FAISS + BM25</span></span>
        </div>
      `;
      sourcesContainer.appendChild(card);
    });
  }



  // A light, clean, regex-based Markdown renderer matching styling design tags
  function parseMarkdown(md) {
    if (!md) return "<p>No output generated.</p>";

    // Remove carriage returns to normalize line endings across platforms
    let text = md.replace(/\r/g, "");

    // Pre-process list-blockquotes (e.g. "* > Quote") to be simple blockquotes
    // This prevents bullet-blockquote rendering conflicts and multiple asterisks parsing bugs
    text = text.replace(/^[ \t]*[*+-][ \t]+>\s+/gm, "> ");

    // Escape basic script tags to prevent HTML injection, keeping spaces
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Replace specific section headers with stylized manuscript headers before general bold processing
    html = html.replace(/(?:^|\n)\s*(?:\*\*)?(?:1\.|I\.)?\s*\*?\*?(?:उत्तर\s*[\/&]\s*Synthesis|Synthesis\s*[\/&]\s*उत्तर):?\s*(?:\*\*)?/gi, '\n<section-marker type="I"></section-marker>');
    html = html.replace(/(?:^|\n)\s*(?:\*\*)?(?:2\.|II\.)?\s*\*?\*?(?:प्रमाण\s*[\/&]\s*Evidence|Evidence\s*[\/&]\s*प्रमाण):?\s*(?:\*\*)?/gi, '\n<section-marker type="II"></section-marker>');
    html = html.replace(/(?:^|\n)\s*(?:\*\*)?(?:3\.|III\.)?\s*\*?\*?(?:संदर्भ\s*[\/&]\s*Context(?:\s*\(Optional\))?|Context(?:\s*\(Optional\))?\s*[\/&]\s*संदर्भ):?\s*(?:\*\*)?/gi, '\n<section-marker type="III"></section-marker>');

    // Parse Blockquotes first
    html = html.replace(/^\s*&gt;\s+(.*$)/gim, "<blockquote>$1</blockquote>");

    // Bold & Italics
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

    // Inline Code
    html = html.replace(/`(.*?)`/g, "<code>$1</code>");

    // Headings
    html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // Paragraphs, Lists & Custom Card Container Wrappers
    const lines = html.split("\n");
    let inList = false;
    let inCard = false;
    let renderedLines = [];

    lines.forEach(line => {
      const trimmed = line.trim();
      
      // If we hit a section header, close any open list or card first
      if (trimmed.startsWith("<section-marker")) {
        if (inList) {
          renderedLines.push("</ul>");
          inList = false;
        }
        if (inCard) {
          renderedLines.push("</div></div>");
          inCard = false;
        }
        
        // Open a new card container depending on which header it is
        if (trimmed.includes('type="I"')) {
          renderedLines.push('<div class="synthesis-card"><div class="synthesis-terminal-header"><div class="terminal-dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div><span class="terminal-title">SYNTHESIS</span></div><div class="synthesis-terminal-body">');
          inCard = true;
        } else if (trimmed.includes('type="II"')) {
          renderedLines.push('<div class="evidence-section-flat"><h3 class="evidence-flat-header">II &nbsp; प्रमाण &bull; Evidence</h3><div class="evidence-flat-body">');
          inCard = true;
        } else if (trimmed.includes('type="III"')) {
          renderedLines.push('<div class="context-section-flat"><h3 class="context-flat-header">III &nbsp; संदर्भ &bull; Context</h3><div class="context-flat-body">');
          inCard = true;
        }
        return;
      }
      
      // Check for list item
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        if (!inList) {
          renderedLines.push("<ul>");
          inList = true;
        }
        renderedLines.push(`<li>${trimmed.substring(2)}</li>`);
      } else {
        if (inList) {
          renderedLines.push("</ul>");
          inList = false;
        }
        
        if (trimmed) {
          // If already a structural block, don't wrap in p tag
          if (trimmed.startsWith("<blockquote") || trimmed.startsWith("<ul") || trimmed.startsWith("<ol") || trimmed.startsWith("</ul") || trimmed.startsWith("</ol")) {
            renderedLines.push(line);
          } else {
            renderedLines.push(`<p>${line}</p>`);
          }
        }
      }
    });

    if (inList) {
      renderedLines.push("</ul>");
    }
    if (inCard) {
      renderedLines.push("</div></div>");
    }

    return renderedLines.join("\n");
  }

  // Manage Query History
  function addToHistory(query) {
    // Filter duplicates
    searchHistory = searchHistory.filter(q => q !== query);
    searchHistory.unshift(query);
    
    // Keep max 10
    if (searchHistory.length > 10) {
      searchHistory.pop();
    }
    
    localStorage.setItem("bhairav_history", JSON.stringify(searchHistory));
    renderHistory();
  }

  function renderHistory() {
    queryHistoryContainer.innerHTML = "";
    
    if (searchHistory.length === 0) {
      queryHistoryContainer.innerHTML = `<div class="empty-history">No recent queries in this session.</div>`;
      return;
    }

    searchHistory.forEach(q => {
      const item = document.createElement("div");
      item.className = "history-item";
      item.textContent = q;
      item.title = q;
      item.addEventListener("click", () => {
        queryInput.value = q;
        executeSearch(q);
      });
      queryHistoryContainer.appendChild(item);
    });
  }
});
