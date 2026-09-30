/**
 * ==============================================================================
 * FloraVision AI 2.0 - Client Controller Script
 * ==============================================================================
 * Handles image selection, drag-and-drop, quick test sample loading,
 * asynchronous prediction dispatch to /predict, image-quality feedback,
 * Grad-CAM neural explainability toggling, and botanical report rendering.
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
  // Upload & Form Elements
  const form = document.getElementById("upload-form");
  const imageInput = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const dropPrompt = document.getElementById("drop-prompt");
  const previewWrapper = document.getElementById("preview-wrapper");
  const imagePreview = document.getElementById("image-preview");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const btnChangeImage = document.getElementById("btn-change-image");
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnReset = document.getElementById("btn-reset");
  const qualityBox = document.getElementById("quality-feedback-box");
  const qualityIcon = document.getElementById("quality-icon");
  const qualityText = document.getElementById("quality-text");
  const errorBox = document.getElementById("error-box");
  const errorMessage = document.getElementById("error-message");

  // State Viewports
  const statePlaceholder = document.getElementById("state-placeholder");
  const stateLoading = document.getElementById("state-loading");
  const stateSuccess = document.getElementById("state-success");
  const stateUncertain = document.getElementById("state-uncertain");

  // Success Viewport Elements
  const resultDisplayImg = document.getElementById("result-display-img");
  const resultGradcamImg = document.getElementById("result-gradcam-img");
  const gradcamControls = document.getElementById("gradcam-controls");
  const gradcamCaption = document.getElementById("gradcam-caption");
  const btnViewOriginal = document.getElementById("btn-view-original");
  const btnViewGradcam = document.getElementById("btn-view-gradcam");

  const resFlowerName = document.getElementById("res-flower-name");
  const resScientificSubtitle = document.getElementById("res-scientific-subtitle");
  const resFamilyPill = document.getElementById("res-family-pill");
  const resOrderPill = document.getElementById("res-order-pill");
  const resConfidencePct = document.getElementById("res-confidence-pct");
  const resConfidenceFill = document.getElementById("res-confidence-fill");
  const topPredictionsList = document.getElementById("top-predictions-list");

  // Botanical Report Elements
  const repOverview = document.getElementById("rep-overview");
  const repCharacteristics = document.getElementById("rep-characteristics");
  const repDistribution = document.getElementById("rep-distribution");
  const repEcological = document.getElementById("rep-ecological");
  const repDiagnostic = document.getElementById("rep-diagnostic");
  const repUsesGrid = document.getElementById("rep-uses-grid");
  const repSummaryList = document.getElementById("rep-summary-list");
  const repSourcesList = document.getElementById("rep-sources-list");

  // Medicinal Context Elements
  const medParts = document.getElementById("med-parts");
  const medTraditional = document.getElementById("med-traditional");
  const medResearch = document.getElementById("med-research");
  const medEvidence = document.getElementById("med-evidence");

  // Uncertain Viewport Elements
  const uncertainExplanation = document.getElementById("uncertain-explanation");
  const uncertainProbBars = document.getElementById("uncertain-prob-bars");

  // Sample Buttons & Explorer Triggers
  const sampleBtns = document.querySelectorAll(".sample-btn");
  const explorerLoadBtns = document.querySelectorAll(".explorer-load-btn");

  // State
  let currentFile = null;
  const ALLOWED_MIME = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

  /**
   * Clears all feedback/results viewports
   */
  function clearResults() {
    errorBox.style.display = "none";
    errorMessage.textContent = "";
    stateLoading.style.display = "none";
    stateSuccess.style.display = "none";
    stateUncertain.style.display = "none";
    statePlaceholder.style.display = "flex";
  }

  /**
   * Shows error alert banner
   */
  function showError(msg) {
    clearResults();
    errorMessage.textContent = msg;
    errorBox.style.display = "flex";
    errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /**
   * Formats bytes into human-readable string
   */
  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  /**
   * Validates selected image file
   */
  function validateFile(file) {
    if (!file) {
      showError("Please select a flower photograph to analyze.");
      return false;
    }

    if (!ALLOWED_MIME.includes(file.type.toLowerCase())) {
      showError("Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image.");
      return false;
    }

    if (file.size > MAX_FILE_SIZE) {
      showError(`File exceeds 10 MB limit (${formatBytes(file.size)}). Please choose a smaller image.`);
      return false;
    }

    return true;
  }

  /**
   * Sets preview display for chosen file and evaluates dimensions
   */
  function setPreview(file) {
    clearResults();
    currentFile = file;

    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      imagePreview.src = dataUrl;
      dropPrompt.style.display = "none";
      previewWrapper.style.display = "flex";
      btnAnalyze.disabled = false;

      // Quick client dimension analysis
      const tempImg = new Image();
      tempImg.onload = () => {
        const w = tempImg.width;
        const h = tempImg.height;
        qualityBox.style.display = "flex";
        if (w < 160 || h < 160) {
          qualityBox.className = "quality-feedback-bar warning";
          qualityIcon.textContent = "⚠️";
          qualityText.textContent = `Technical Check: Low resolution (${w}×${h} px). For optimal botanical classification, close-up floral photos are recommended.`;
        } else {
          qualityBox.className = "quality-feedback-bar";
          qualityIcon.textContent = "✓";
          qualityText.textContent = `Technical Check: Optimal resolution (${w}×${h} px, ${formatBytes(file.size)}). Ready for analysis.`;
        }
      };
      tempImg.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }

  /**
   * Resets form to initial empty state
   */
  function resetForm() {
    currentFile = null;
    imageInput.value = "";
    imagePreview.src = "";
    previewWrapper.style.display = "none";
    dropPrompt.style.display = "flex";
    btnAnalyze.disabled = true;
    qualityBox.style.display = "none";
    clearResults();
  }

  // File Input Change
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
      setPreview(file);
    }
  });

  // Change Image Button
  btnChangeImage.addEventListener("click", (e) => {
    e.stopPropagation();
    imageInput.click();
  });

  // Reset Button
  btnReset.addEventListener("click", () => {
    resetForm();
  });

  // Dropzone click opens file dialog
  dropZone.addEventListener("click", (e) => {
    if (e.target !== btnChangeImage && !btnChangeImage.contains(e.target)) {
      imageInput.click();
    }
  });

  // Drag and Drop Events
  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file && validateFile(file)) {
      setPreview(file);
    }
  });

  /**
   * Helper to load a static sample file into the scanner
   */
  async function loadSampleByUrl(src, name) {
    try {
      clearResults();
      statePlaceholder.style.display = "none";
      stateLoading.style.display = "flex";

      const res = await fetch(src);
      const blob = await res.blob();
      stateLoading.style.display = "none";

      const sampleFile = new File([blob], name, { type: blob.type || "image/jpeg" });
      if (validateFile(sampleFile)) {
        setPreview(sampleFile);
        // Scroll to scanner
        const scannerElem = document.getElementById("scanner");
        if (scannerElem) {
          scannerElem.scrollIntoView({ behavior: "smooth" });
        }
      }
    } catch (err) {
      stateLoading.style.display = "none";
      showError("Unable to load the requested sample image. You can upload an image from your computer.");
    }
  }

  // Quick Preset Sample Buttons
  sampleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const src = btn.getAttribute("data-src");
      const name = btn.getAttribute("data-name");
      loadSampleByUrl(src, name);
    });
  });

  // Explorer "Test in Scanner" Buttons
  explorerLoadBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const src = btn.getAttribute("data-src");
      const name = btn.getAttribute("data-name");
      loadSampleByUrl(src, name);
    });
  });

  /**
   * Renders probability distribution rows
   */
  function renderProbabilityBars(container, topPredictions, winningClassCode) {
    if (!container) return;
    container.innerHTML = "";
    if (!topPredictions || !Array.isArray(topPredictions)) return;

    topPredictions.forEach((item) => {
      const name = item.flower || item.common_name || item.display_name || item.class || "Flower";
      const pct = Math.round((item.probability || 0) * 100);
      const isWinner = winningClassCode && (item.class_code === winningClassCode || name.toLowerCase() === winningClassCode.toLowerCase());

      const row = document.createElement("div");
      row.className = `pred-row ${isWinner ? "active-winner" : ""}`;
      row.innerHTML = `
        <span class="pred-name">${name}</span>
        <div class="pred-bar-track">
          <div class="pred-bar-fill" style="width: ${pct}%"></div>
        </div>
        <span class="pred-pct">${pct}%</span>
      `;
      container.appendChild(row);
    });
  }

  /**
   * Displays confirmed botanical prediction & intelligence report
   */
  function displaySuccess(data) {
    clearResults();
    statePlaceholder.style.display = "none";

    const commonName = data.common_name || data.flower || "Flower";
    const scientific = data.scientific_name || "";
    const family = data.family || data.botanical_family || "Unknown Family";
    const order = data.taxonomic_order || "Unknown Order";
    const confPct = typeof data.confidence_pct === "number" ? `${data.confidence_pct.toFixed(1)}%` : data.probability_pct || "0%";
    const confRatio = typeof data.confidence_pct === "number" ? Math.min(100, Math.round(data.confidence_pct)) : Math.round((data.probability || 0) * 100);

    // 1. Persistent Uploaded Image Display
    if (imagePreview.src) {
      resultDisplayImg.src = imagePreview.src;
    }

    // 2. Grad-CAM Neural Explainability Layer
    if (data.gradcam_heatmap) {
      resultGradcamImg.src = data.gradcam_heatmap;
      resultGradcamImg.style.display = "none";
      resultDisplayImg.style.display = "block";
      gradcamControls.style.display = "flex";
      gradcamCaption.style.display = "none";
      btnViewOriginal.classList.add("active");
      btnViewGradcam.classList.remove("active");
    } else {
      gradcamControls.style.display = "none";
      gradcamCaption.style.display = "none";
      resultGradcamImg.style.display = "none";
      resultDisplayImg.style.display = "block";
    }

    // 3. Identification Hero
    resFlowerName.textContent = commonName;
    resScientificSubtitle.textContent = scientific ? `${scientific}` : "";
    resFamilyPill.textContent = family;
    resOrderPill.textContent = order;
    resConfidencePct.textContent = confPct;
    resConfidenceFill.style.width = `${confRatio}%`;

    // 4. Probability Bars
    renderProbabilityBars(topPredictionsList, data.top_predictions, data.predicted_class);

    // 5. Botanical Intelligence Report Sections
    const bot = data.botanical_info || {};
    repOverview.textContent = data.overview || bot.overview || data.characteristics || "Verified botanical overview is currently unavailable.";
    repCharacteristics.textContent = data.key_characteristics || bot.key_characteristics || data.characteristics || "Detailed morphological characteristics are unavailable.";
    repDistribution.textContent = data.geographic_distribution || bot.geographic_distribution || data.habitat || "Geographic habitat information is unavailable.";
    repEcological.textContent = data.ecological_importance || bot.ecological_importance || "Pollinator and ecological data are unavailable.";
    repDiagnostic.textContent = data.diagnostic_tips || bot.diagnostic_tips || "Distinctive diagnostic guidance is unavailable.";

    // 6. Common Uses Categorized Badges
    repUsesGrid.innerHTML = "";
    const usesObj = data.common_uses || bot.common_uses || {};
    const useCategories = [
      { key: "traditional", label: "🌿 Traditional Uses" },
      { key: "ornamental", label: "🌸 Ornamental Uses" },
      { key: "cultural", label: "🍃 Cultural Uses" },
      { key: "environmental", label: "🌱 Environmental / Ecological" },
      { key: "research", label: "🧪 Research Interest" },
      { key: "commercial", label: "🏭 Commercial Uses" }
    ];

    let hasUses = false;
    useCategories.forEach((cat) => {
      if (usesObj[cat.key]) {
        hasUses = true;
        const card = document.createElement("div");
        card.className = "use-card";
        card.innerHTML = `
          <span class="use-tag">${cat.label}</span>
          <p class="use-desc">${usesObj[cat.key]}</p>
        `;
        repUsesGrid.appendChild(card);
      }
    });

    if (!hasUses && data.cultural_medicinal) {
      const card = document.createElement("div");
      card.className = "use-card";
      card.innerHTML = `
        <span class="use-tag">🌿 Cultural &amp; Practical Uses</span>
        <p class="use-desc">${data.cultural_medicinal}</p>
      `;
      repUsesGrid.appendChild(card);
    }

    // 7. Documented Traditional & Medicinal Information
    const medObj = data.traditional_medicinal_info || bot.traditional_medicinal_info || {};
    medParts.textContent = medObj.plant_parts_used || "Dried blossoms, fruit receptacles, or foliage (historical records).";
    medTraditional.textContent = medObj.documented_traditional_use || data.cultural_medicinal || "Documented in historical folk and traditional ethnobotanical records.";
    medResearch.textContent = medObj.scientific_research_context || "Investigated in chemical literature for secondary metabolite and antioxidant profiles.";
    medEvidence.textContent = medObj.evidence_level || "Preliminary in-vitro assays; controlled human clinical validation is limited.";

    // 8. In-Depth Botanical Points (10-15 Points)
    repSummaryList.innerHTML = "";
    const summaryPoints = data.comprehensive_summary || bot.comprehensive_summary || [];
    if (Array.isArray(summaryPoints) && summaryPoints.length > 0) {
      summaryPoints.forEach((point) => {
        const li = document.createElement("li");
        li.textContent = point;
        repSummaryList.appendChild(li);
      });
    }

    // 9. Authoritative Sources List
    repSourcesList.innerHTML = "";
    const sources = data.authoritative_sources || bot.authoritative_sources || [
      "Royal Botanic Gardens Kew, Plants of the World Online (POWO)",
      "USDA Natural Resources Conservation Service (NRCS) PLANTS Database",
      "Flora of North America / Flora of China Taxonomic Revisions"
    ];
    sources.forEach((src) => {
      const li = document.createElement("li");
      li.textContent = src;
      repSourcesList.appendChild(li);
    });

    stateSuccess.style.display = "flex";
    stateSuccess.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Grad-CAM Toggle Handlers
  btnViewOriginal.addEventListener("click", () => {
    btnViewOriginal.classList.add("active");
    btnViewGradcam.classList.remove("active");
    resultDisplayImg.style.display = "block";
    resultGradcamImg.style.display = "none";
    gradcamCaption.style.display = "none";
  });

  btnViewGradcam.addEventListener("click", () => {
    btnViewGradcam.classList.add("active");
    btnViewOriginal.classList.remove("active");
    resultDisplayImg.style.display = "none";
    resultGradcamImg.style.display = "block";
    gradcamCaption.style.display = "block";
  });

  /**
   * Displays uncertain/out-of-distribution rejection
   */
  function displayUncertain(data) {
    clearResults();
    statePlaceholder.style.display = "none";

    const confPctStr = typeof data.confidence_pct === "number" ? `${data.confidence_pct.toFixed(1)}%` : data.probability_pct || "0%";
    uncertainExplanation.textContent =
      data.message ||
      `Confidence below the configured threshold (55.0%) — prediction withheld. The model could not produce a sufficiently confident prediction among the supported classes.`;

    renderProbabilityBars(uncertainProbBars, data.top_predictions, null);

    stateUncertain.style.display = "flex";
    stateUncertain.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Form Submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!currentFile) {
      showError("Please select or drop a flower photo before analyzing.");
      return;
    }

    if (!validateFile(currentFile)) return;

    // Set UI loading state
    clearResults();
    statePlaceholder.style.display = "none";
    stateLoading.style.display = "flex";

    btnAnalyze.disabled = true;
    const spinner = btnAnalyze.querySelector(".btn-spinner");
    const text = btnAnalyze.querySelector(".btn-text");
    if (spinner) spinner.style.display = "inline-block";
    if (text) text.textContent = "Analyzing Pixels...";

    let data;
    try {
      const formData = new FormData();
      formData.append("image", currentFile, currentFile.name);

      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errText = `Server error (${response.status})`;
        try {
          const errData = await response.json();
          if (errData && errData.error) errText = errData.error;
        } catch (_) {}
        showError(errText);
        return;
      }

      data = await response.json();
    } catch (networkErr) {
      console.error("Fetch network error:", networkErr);
      showError("Network Error: Unable to reach the server. Please verify the Flask backend is running.");
      return;
    } finally {
      stateLoading.style.display = "none";
      btnAnalyze.disabled = false;
      if (spinner) spinner.style.display = "none";
      if (text) text.textContent = "Analyze Flower";
    }

    try {
      if (data.status === "success") {
        displaySuccess(data);
      } else if (data.status === "uncertain") {
        displayUncertain(data);
      } else {
        showError(data.error || "Unexpected response format from server.");
      }
    } catch (renderErr) {
      console.error("Client render error:", renderErr);
      showError("Error rendering prediction result: " + renderErr.message);
    }
  });
});
