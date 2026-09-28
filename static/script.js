/**
 * ==============================================================================
 * FloraVision AI - Modern Client-Side JavaScript
 * ==============================================================================
 * Handles drag-and-drop, sample thumbnail gallery interaction, file validation,
 * image preview, asynchronous FormData submission to Flask /predict, and dynamic
 * rendering of deep learning predictions and botanical metadata.
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
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

  // Feedback & State Views
  const placeholderBox = document.getElementById("placeholder-box");
  const loadingBox = document.getElementById("loading-box");
  const successBox = document.getElementById("result-success-box");
  const uncertainBox = document.getElementById("result-uncertain-box");
  const errorBox = document.getElementById("error-box");
  const errorMessage = document.getElementById("error-message");

  // Success Card Elements
  const resSpeciesName = document.getElementById("res-species-name");
  const resScientificName = document.getElementById("res-scientific-name");
  const resConfidencePct = document.getElementById("res-confidence-pct");
  const resConfidenceSub = document.getElementById("res-confidence-sub");
  const resConfidenceBar = document.getElementById("res-confidence-bar");
  const resBadge = document.getElementById("res-badge");
  const bFamily = document.getElementById("b-family");
  const bOrder = document.getElementById("b-order");
  const bFullTaxonomy = document.getElementById("b-full-taxonomy");
  const bCommonNames = document.getElementById("b-common-names");
  const bCharacteristics = document.getElementById("b-characteristics");
  const bHabitat = document.getElementById("b-habitat");
  const bCultural = document.getElementById("b-cultural");
  const bDiagnostic = document.getElementById("b-diagnostic");
  const bSummaryList = document.getElementById("b-summary-list");
  const probList = document.getElementById("prob-list");

  // Uncertain Card Elements
  const uncertainMessage = document.getElementById("uncertain-message");
  const uncertainProbList = document.getElementById("uncertain-prob-list");

  // Sample Cards
  const sampleCards = document.querySelectorAll(".sample-card");

  // State
  let currentFile = null;
  const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 Megabytes

  /**
   * Clears all feedback/result panes to neutral state
   */
  function clearResults() {
    errorBox.style.display = "none";
    errorMessage.textContent = "";
    loadingBox.style.display = "none";
    successBox.style.display = "none";
    uncertainBox.style.display = "none";
    placeholderBox.style.display = "flex";
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
      showError("Please select a flower image to analyze.");
      return false;
    }

    if (!ALLOWED_TYPES.includes(file.type.toLowerCase())) {
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
   * Sets preview display for chosen file
   */
  function setPreview(file) {
    clearResults();

    currentFile = file;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropPrompt.style.display = "none";
      previewWrapper.style.display = "flex";
      btnAnalyze.disabled = false;
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
    sampleCards.forEach((c) => c.classList.remove("active"));
    clearResults();
  }

  // File Input Change
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
      sampleCards.forEach((c) => c.classList.remove("active"));
      setPreview(file);
    }
  });

  // Change Photo button
  btnChangeImage.addEventListener("click", (e) => {
    e.stopPropagation();
    imageInput.click();
  });

  // Reset button
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
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file && validateFile(file)) {
      sampleCards.forEach((c) => c.classList.remove("active"));
      setPreview(file);
    }
  });

  // Quick Preset Sample Photo Cards
  sampleCards.forEach((card) => {
    card.addEventListener("click", async () => {
      sampleCards.forEach((c) => c.classList.remove("active"));
      card.classList.add("active");

      const src = card.getAttribute("data-src");
      const name = card.getAttribute("data-name");

      try {
        clearResults();
        placeholderBox.style.display = "none";
        loadingBox.style.display = "flex";

        const res = await fetch(src);
        const blob = await res.blob();
        loadingBox.style.display = "none";

        const sampleFile = new File([blob], name, { type: blob.type || "image/jpeg" });
        if (validateFile(sampleFile)) {
          setPreview(sampleFile);
        }
      } catch (err) {
        loadingBox.style.display = "none";
        showError("Failed to load sample photo. You can upload an image from your computer.");
      }
    });
  });

  /**
   * Renders probability distribution rows
   */
  function renderProbabilities(container, topPredictions, winningClass) {
    if (!container) return;
    container.innerHTML = "";
    if (!topPredictions || !Array.isArray(topPredictions)) return;

    topPredictions.forEach((item) => {
      const clsName = item.flower || item.common_name || item.class || item.display_name || item.class_code || "Flower";
      const pct = Math.round((item.probability || 0) * 100);
      const isWinner = winningClass && clsName.toLowerCase() === winningClass.toLowerCase();

      const row = document.createElement("div");
      row.className = `prob-row ${isWinner ? "winner" : ""}`;
      row.innerHTML = `
        <span class="prob-name">${clsName}</span>
        <div class="prob-track">
          <div class="prob-fill ${isWinner ? "winner" : ""}" style="width: ${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      `;
      container.appendChild(row);
    });
  }

  /**
   * Displays confirmed botanical prediction
   */
  function displaySuccess(data) {
    clearResults();
    if (placeholderBox) placeholderBox.style.display = "none";

    const commonName = data.common_name || data.predicted_class || data.flower || "Flower";
    const scientific = data.scientific_name || "";
    const confPct = typeof data.confidence_pct === "number" ? `${data.confidence_pct.toFixed(1)}%` : data.probability_pct || "0%";
    const confRatio = typeof data.confidence_pct === "number" ? Math.min(100, Math.round(data.confidence_pct)) : Math.round((data.probability || 0) * 100);

    if (resSpeciesName) resSpeciesName.textContent = commonName;
    if (resScientificName) resScientificName.textContent = scientific ? `${scientific}` : "";
    if (resConfidencePct) resConfidencePct.textContent = confPct;
    if (resConfidenceSub) resConfidenceSub.textContent = `${confPct} probability`;
    if (resConfidenceBar) resConfidenceBar.style.width = `${confRatio}%`;

    // Botanical Taxonomy & Hierarchy
    if (data.botanical_family && bFamily) {
      bFamily.textContent = data.botanical_family;
    }
    if (data.taxonomic_order && bOrder) {
      bOrder.textContent = data.taxonomic_order;
    }
    if (bFullTaxonomy) {
      bFullTaxonomy.textContent = data.full_taxonomy || (data.botanical_info && data.botanical_info.full_taxonomy) || "";
    }
    if (bCommonNames) {
      bCommonNames.textContent = data.common_names || (data.botanical_info && data.botanical_info.common_names) || "";
    }
    if (bCharacteristics) {
      bCharacteristics.textContent = data.characteristics || (data.botanical_info && data.botanical_info.characteristics) || "";
    }
    if (bHabitat) {
      bHabitat.textContent = data.habitat || (data.botanical_info && data.botanical_info.habitat) || "";
    }
    if (bCultural) {
      bCultural.textContent = data.cultural_medicinal || (data.botanical_info && data.botanical_info.cultural_medicinal) || "";
    }
    if (bDiagnostic) {
      bDiagnostic.textContent = data.diagnostic_tips || (data.botanical_info && data.botanical_info.diagnostic_tips) || "";
    }

    // Populate the 10 to 15 In-Depth Bullet Points
    if (bSummaryList) {
      bSummaryList.innerHTML = "";
      const summaryPoints = data.comprehensive_summary || (data.botanical_info && data.botanical_info.comprehensive_summary) || [];
      if (Array.isArray(summaryPoints) && summaryPoints.length > 0) {
        summaryPoints.forEach((point) => {
          const li = document.createElement("li");
          li.textContent = point;
          bSummaryList.appendChild(li);
        });
      }
    }

    // Probability breakdown
    renderProbabilities(probList, data.top_predictions, commonName);

    if (successBox) {
      successBox.style.display = "flex";
      successBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  /**
   * Displays uncertain/out-of-distribution rejection
   */
  function displayUncertain(data) {
    clearResults();
    if (placeholderBox) placeholderBox.style.display = "none";

    const confPctStr = typeof data.confidence_pct === "number" ? `${data.confidence_pct.toFixed(1)}%` : data.probability_pct || "0%";

    if (uncertainMessage) {
      uncertainMessage.textContent =
        data.message ||
        `The model's highest probability (${confPctStr}) is below the conservative 55.0% threshold. Botanical taxonomy is suppressed.`;
    }

    renderProbabilities(uncertainProbList, data.top_predictions, null);

    if (uncertainBox) {
      uncertainBox.style.display = "flex";
      uncertainBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
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
    if (placeholderBox) placeholderBox.style.display = "none";
    if (loadingBox) loadingBox.style.display = "flex";
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
      showError(
        "Network Error: Unable to reach the Flask server. Please verify that 'python app.py' is running on http://127.0.0.1:5000."
      );
      return;
    } finally {
      if (loadingBox) loadingBox.style.display = "none";
      btnAnalyze.disabled = false;
      if (spinner) spinner.style.display = "none";
      if (text) text.textContent = "Analyze Flower with CNN";
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
