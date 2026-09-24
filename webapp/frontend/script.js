/**
 * AI Grooming Assistant — Next-Gen Frontend Prototype Script
 * Clean ES6+ Vanilla JavaScript implementation (No external frameworks)
 */

// ==========================================================================
// 1. APPLICATION STATE MANAGEMENT
// ==========================================================================
const state = {
  photoFile: null,
  photoUrl: null,
  gender: null,          // 'men' | 'women'
  occasion: null,        // 'casual' | 'formal' | 'party'
  isAnalyzing: false,
  analysisResult: null,
  selectedGlasses: null,
  allGlasses: [],        // every frame the backend serves: [{id, styles, image}]
  tryonIndex: 0,         // which of allGlasses is on the face right now
  tryonLeft: 50,         // Default X %
  tryonTop: 33,          // Default Y % (eye level)
  tryonScale: 42,        // Default scale % (realistic face proportion)
  isDraggingGlasses: false
};

// ==========================================================================
// 2. STATIC UI COPY
// (Descriptive text only — actual recommendation logic/data always comes
// from the backend, which reads it from recommendations.py, so it can
// never drift out of sync the way a second hardcoded copy could.)
// ==========================================================================

/**
 * Face Shape Descriptions
 */
const faceShapeDetails = {
  Oval: "Balanced proportions with a softly curved jawline and slightly wider cheekbones. Extremely versatile for frame styles.",
  Square: "Strong angular jawline with proportional forehead and cheekbone width. Best paired with curved or round frames.",
  Round: "Soft curved contours with equal face width and length. Complemented by geometric rectangular frames for added structure.",
  Heart: "Broader forehead tapering down smoothly to a narrow chin. Suits bottom-heavy or aviator frames.",
  Oblong: "Longer face with a fairly straight cheek line. Suits frames and styles that add width rather than height."
};

/**
 * Skin Tone Descriptions, keyed by the categories the backend's
 * skin_tone.py actually classifies into.
 */
const skinToneDetails = {
  "Very Light": "Fair undertones — pastel and jewel tones tend to stand out beautifully.",
  "Light": "Light undertones — earthy and muted colors work especially well.",
  "Medium": "Balanced medium undertones — most jewel tones and warm neutrals suit you.",
  "Tan": "Warm tan undertones — bold, saturated colors bring out your natural warmth.",
  "Brown": "Rich brown undertones — bright colors and warm metallics create striking contrast.",
  "Deep": "Deep, rich undertones — high-contrast bold colors and warm golds shine brightest."
};

// ==========================================================================
// 3. DOM ELEMENTS REFERENCE
// ==========================================================================
let DOM = {};

function cacheDOMElements() {
  DOM = {
    groomingForm: document.getElementById("groomingForm"),
    uploadScreen: document.getElementById("uploadScreen"),
    loadingState: document.getElementById("loadingState"),
    resultsScreen: document.getElementById("resultsScreen"),
    
    // Inputs & Options
    genderOptions: document.getElementById("genderOptions"),
    occasionOptions: document.getElementById("occasionOptions"),
    photoInput: document.getElementById("photoInput"),
    uploadDropzone: document.getElementById("uploadDropzone"),
    previewContainer: document.getElementById("previewContainer"),
    photoPreview: document.getElementById("photoPreview"),
    fileNameBadge: document.getElementById("fileNameBadge"),
    changePhotoBtn: document.getElementById("changePhotoBtn"),
    
    // Actions & Feedback
    errorMessage: document.getElementById("errorMessage"),
    errorText: document.getElementById("errorText"),
    analyzeButton: document.getElementById("analyzeButton"),
    scannerImg: document.getElementById("scannerImg"),
    loadingStepText: document.getElementById("loadingStepText"),
    tryAnotherButton: document.getElementById("tryAnotherButton"),
    
    // Results Elements
    resultsPhoto: document.getElementById("resultsPhoto"),
    faceShape: document.getElementById("faceShape"),
    faceShapeDesc: document.getElementById("faceShapeDesc"),
    skinTone: document.getElementById("skinTone"),
    skinToneDesc: document.getElementById("skinToneDesc"),
    skinToneSwatchCircle: document.getElementById("skinToneSwatchCircle"),
    skinToneLowConfidenceWarning: document.getElementById("skinToneLowConfidenceWarning"),
    confidence: document.getElementById("confidence"),
    
    // Try-On Stage Elements & Sliders
    tryonContainer: document.getElementById("tryonContainer"),
    tryonFace: document.getElementById("tryonFace"),
    tryonGlassesOverlay: document.getElementById("tryonGlassesOverlay"),
    tryonGlassesTitle: document.getElementById("tryonGlassesTitle"),
    tryonGlassesReason: document.getElementById("tryonGlassesReason"),
    tryonPrevBtn: document.getElementById("tryonPrevBtn"),
    tryonNextBtn: document.getElementById("tryonNextBtn"),
    tryonCounter: document.getElementById("tryonCounter"),
    tryonXSlider: document.getElementById("tryonXSlider"),
    tryonHeightSlider: document.getElementById("tryonHeightSlider"),
    tryonScaleSlider: document.getElementById("tryonScaleSlider"),
    presetDefaultBtn: document.getElementById("presetDefaultBtn"),
    presetHigherBtn: document.getElementById("presetHigherBtn"),
    presetLowerBtn: document.getElementById("presetLowerBtn"),
    
    // Recommendation Containers
    glassesGrid: document.getElementById("glassesGrid"),
    colorSwatchesGrid: document.getElementById("colorSwatchesGrid"),
    avoidColorsGrid: document.getElementById("avoidColorsGrid"),
    colorExplanationText: document.getElementById("colorExplanationText"),
    hairstyleList: document.getElementById("hairstyleList"),
    groomingList: document.getElementById("groomingList")
  };
}

// ==========================================================================
// 4. INITIALIZATION & EVENT LISTENERS
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  cacheDOMElements();
  setupEventListeners();
  setupGlassesDragAndDrop();
});

function setupEventListeners() {
  // Form submission (Analyze CTA)
  if (DOM.groomingForm) {
    DOM.groomingForm.addEventListener("submit", handleFormSubmit);
  }

  // Gender Buttons Selection
  if (DOM.genderOptions) {
    DOM.genderOptions.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-gender]");
      if (btn) {
        selectGender(btn.dataset.gender);
      }
    });
  }

  // Occasion Buttons Selection
  if (DOM.occasionOptions) {
    DOM.occasionOptions.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-occasion]");
      if (btn) {
        selectOccasion(btn.dataset.occasion);
      }
    });
  }

  // File Upload Handlers (Click & Drag-Drop)
  if (DOM.uploadDropzone && DOM.photoInput) {
    DOM.uploadDropzone.addEventListener("click", () => DOM.photoInput.click());
    DOM.photoInput.addEventListener("change", handleFileSelect);
    
    // Drag and Drop Events
    DOM.uploadDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      DOM.uploadDropzone.classList.add("drag-active");
    });
    
    DOM.uploadDropzone.addEventListener("dragleave", () => {
      DOM.uploadDropzone.classList.remove("drag-active");
    });
    
    DOM.uploadDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      DOM.uploadDropzone.classList.remove("drag-active");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });
  }

  // Change Photo Button
  if (DOM.changePhotoBtn) {
    DOM.changePhotoBtn.addEventListener("click", () => {
      DOM.photoInput.click();
    });
  }

  // Browse every available frame
  if (DOM.tryonPrevBtn) DOM.tryonPrevBtn.addEventListener("click", () => stepGlasses(-1));
  if (DOM.tryonNextBtn) DOM.tryonNextBtn.addEventListener("click", () => stepGlasses(1));

  // Try-On Alignment Sliders Sync
  if (DOM.tryonXSlider) {
    DOM.tryonXSlider.addEventListener("input", (e) => {
      state.tryonLeft = parseFloat(e.target.value);
      updateGlassesTransform();
    });
  }

  if (DOM.tryonHeightSlider) {
    DOM.tryonHeightSlider.addEventListener("input", (e) => {
      state.tryonTop = parseFloat(e.target.value);
      updateGlassesTransform();
    });
  }

  if (DOM.tryonScaleSlider) {
    DOM.tryonScaleSlider.addEventListener("input", (e) => {
      state.tryonScale = parseFloat(e.target.value);
      updateGlassesTransform();
    });
  }

  // Preset Buttons
  if (DOM.presetDefaultBtn) {
    DOM.presetDefaultBtn.addEventListener("click", () => {
      state.tryonLeft = 50;
      state.tryonTop = 33;
      state.tryonScale = 42;
      syncSlidersUI();
      updateGlassesTransform();
    });
  }

  if (DOM.presetHigherBtn) {
    DOM.presetHigherBtn.addEventListener("click", () => {
      state.tryonTop = 26;
      syncSlidersUI();
      updateGlassesTransform();
    });
  }

  if (DOM.presetLowerBtn) {
    DOM.presetLowerBtn.addEventListener("click", () => {
      state.tryonTop = 40;
      syncSlidersUI();
      updateGlassesTransform();
    });
  }

  // Reset Button (Try Another Photo)
  if (DOM.tryAnotherButton) {
    DOM.tryAnotherButton.addEventListener("click", resetApplication);
  }
}

// ==========================================================================
// 5. INTERACTIVE GLASSES DRAG-AND-DROP HANDLER
// ==========================================================================
function setupGlassesDragAndDrop() {
  const overlay = DOM.tryonGlassesOverlay;
  const container = DOM.tryonContainer;
  if (!overlay || !container) return;

  function startDrag(e) {
    e.preventDefault();
    state.isDraggingGlasses = true;
    overlay.classList.add("is-dragging");
  }

  function moveDrag(e) {
    if (!state.isDraggingGlasses) return;

    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    // Calculate relative percentage position inside container
    let percentX = ((clientX - rect.left) / rect.width) * 100;
    let percentY = ((clientY - rect.top) / rect.height) * 100;

    // Clamp values within realistic screen bounds
    percentX = Math.max(20, Math.min(80, percentX));
    percentY = Math.max(10, Math.min(70, percentY));

    state.tryonLeft = Math.round(percentX);
    state.tryonTop = Math.round(percentY);

    syncSlidersUI();
    updateGlassesTransform();
  }

  function endDrag() {
    if (state.isDraggingGlasses) {
      state.isDraggingGlasses = false;
      overlay.classList.remove("is-dragging");
    }
  }

  // Mouse Events
  overlay.addEventListener("mousedown", startDrag);
  window.addEventListener("mousemove", moveDrag);
  window.addEventListener("mouseup", endDrag);

  // Touch Events for Mobile / Touchscreens
  overlay.addEventListener("touchstart", startDrag, { passive: false });
  window.addEventListener("touchmove", moveDrag, { passive: false });
  window.addEventListener("touchend", endDrag);
}

function syncSlidersUI() {
  if (DOM.tryonXSlider) DOM.tryonXSlider.value = state.tryonLeft;
  if (DOM.tryonHeightSlider) DOM.tryonHeightSlider.value = state.tryonTop;
  if (DOM.tryonScaleSlider) DOM.tryonScaleSlider.value = state.tryonScale;
}

// ==========================================================================
// 6. GENDER & OCCASION SELECTION LOGIC
// ==========================================================================
function selectGender(genderVal) {
  state.gender = genderVal;
  clearError();
  
  const buttons = DOM.genderOptions.querySelectorAll("[data-gender]");
  buttons.forEach((btn) => {
    const isSelected = btn.dataset.gender === genderVal;
    btn.classList.toggle("selected", isSelected);
    btn.setAttribute("aria-checked", isSelected ? "true" : "false");
  });
}

function selectOccasion(occasionVal) {
  state.occasion = occasionVal;
  clearError();
  
  const buttons = DOM.occasionOptions.querySelectorAll("[data-occasion]");
  buttons.forEach((btn) => {
    const isSelected = btn.dataset.occasion === occasionVal;
    btn.classList.toggle("selected", isSelected);
    btn.setAttribute("aria-checked", isSelected ? "true" : "false");
  });
}

// ==========================================================================
// 7. PHOTO FILE HANDLING & VALIDATION
// ==========================================================================
function handleFileSelect(e) {
  if (e.target.files && e.target.files.length > 0) {
    processUploadedFile(e.target.files[0]);
  }
}

function processUploadedFile(file) {
  clearError();
  
  // Validate File Type (JPG, JPEG, PNG only)
  const validTypes = ["image/jpeg", "image/png", "image/jpg"];
  if (!validTypes.includes(file.type.toLowerCase())) {
    showError("Please upload a valid JPG or PNG image file.");
    return;
  }

  // Revoke old object URL if exists to prevent memory leaks
  if (state.photoUrl) {
    URL.revokeObjectURL(state.photoUrl);
  }

  state.photoFile = file;
  state.photoUrl = URL.createObjectURL(file);

  // Update UI Elements
  DOM.photoPreview.src = state.photoUrl;
  DOM.fileNameBadge.textContent = file.name;
  DOM.previewContainer.classList.add("active");
  DOM.uploadDropzone.style.display = "none";
}

// ==========================================================================
// 8. FORM VALIDATION & SUBMISSION
// ==========================================================================
function validateForm() {
  const missing = [];
  
  if (!state.gender) missing.push("gender");
  if (!state.occasion) missing.push("occasion");
  if (!state.photoFile) missing.push("photo");

  if (missing.length === 3) {
    showError("Please select a gender, an occasion, and upload a photo to proceed.");
    return false;
  }
  if (!state.gender) {
    showError("Please select a gender option.");
    return false;
  }
  if (!state.occasion) {
    showError("Please select an occasion option.");
    return false;
  }
  if (!state.photoFile) {
    showError("Please upload a front-facing photo first.");
    return false;
  }

  clearError();
  return true;
}

function showError(msg) {
  DOM.errorText.textContent = msg;
  DOM.errorMessage.classList.add("visible");
  DOM.errorMessage.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearError() {
  DOM.errorText.textContent = "";
  DOM.errorMessage.classList.remove("visible");
}

async function handleFormSubmit(e) {
  e.preventDefault();
  
  if (state.isAnalyzing) return; // Prevent repeated clicks
  
  if (!validateForm()) {
    return;
  }

  startAnalysis();
}

// ==========================================================================
// 9. SIMULATED MOCK AI ANALYSIS & SCANNER ANIMATION STATE
// ==========================================================================
async function startAnalysis() {
  state.isAnalyzing = true;
  DOM.analyzeButton.disabled = true;
  
  // Set portrait inside laser scanner stage
  if (state.photoUrl && DOM.scannerImg) {
    DOM.scannerImg.src = state.photoUrl;
  }

  showLoadingScreen();
  
  // Animated scanner step text
  const steps = [
    "Mapping 68 facial landmark coordinates...",
    "Evaluating skin undertones and contrast balance...",
    "Matching eyewear frames and wardrobe color matrices..."
  ];
  
  let stepIndex = 0;
  DOM.loadingStepText.textContent = steps[0];
  const stepInterval = setInterval(() => {
    stepIndex++;
    if (stepIndex < steps.length) {
      DOM.loadingStepText.textContent = steps[stepIndex];
    }
  }, 500);

  try {
    const result = await analyzeImage(state.photoFile, state.gender, state.occasion);
    clearInterval(stepInterval);

    state.analysisResult = result;
    state.isAnalyzing = false;
    DOM.analyzeButton.disabled = false;

    showResultsScreen();
    renderResults(result);
  } catch (err) {
    clearInterval(stepInterval);
    state.isAnalyzing = false;
    DOM.analyzeButton.disabled = false;
    hideLoadingScreen();
    showError(err.message || "An error occurred during analysis. Please try again.");
  }
}

// Backend API base URL. Change this if the Flask server runs somewhere
// other than localhost:5001 (e.g. once it's deployed/hosted).
const API_BASE_URL = "http://localhost:5001";

/**
 * Calls the real backend (webapp/backend/backend_api.py), which runs the
 * same face shape / skin tone / recommendation logic main.py uses.
 */
async function analyzeImage(file, gender, occasion) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('gender', gender);
  formData.append('occasion', occasion);

  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || "Analysis failed. Please try again.");
  }
  return data;
}

// ==========================================================================
// 10. SCREEN SWITCHING (SPA FLOW)
// ==========================================================================
function showUploadScreen() {
  DOM.uploadScreen.style.display = "block";
  DOM.loadingState.classList.remove("active");
  DOM.resultsScreen.classList.remove("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showLoadingScreen() {
  DOM.uploadScreen.style.display = "none";
  DOM.resultsScreen.classList.remove("active");
  DOM.loadingState.classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function hideLoadingScreen() {
  DOM.loadingState.classList.remove("active");
  DOM.uploadScreen.style.display = "block";
}

function showResultsScreen() {
  DOM.loadingState.classList.remove("active");
  DOM.uploadScreen.style.display = "none";
  DOM.resultsScreen.classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ==========================================================================
// 11. DYNAMIC RENDER FUNCTIONS
// ==========================================================================
function renderResults(result) {
  // 1. Render Uploaded Photo
  if (state.photoUrl) {
    DOM.resultsPhoto.src = state.photoUrl;
    DOM.tryonFace.src = state.photoUrl;
  }

  // 2. Render Face Shape
  const shape = result.faceShape || "Oval";
  DOM.faceShape.textContent = shape;
  DOM.faceShapeDesc.textContent = faceShapeDetails[shape] || "";

  // 3. Render Skin Tone
  const tone = result.skinTone || "Medium";
  DOM.skinTone.textContent = tone;
  DOM.skinToneDesc.textContent = skinToneDetails[tone] || "";
  DOM.skinToneSwatchCircle.style.backgroundColor = result.skinToneSwatch || "#c8a27a";
  if (DOM.skinToneLowConfidenceWarning) {
    DOM.skinToneLowConfidenceWarning.style.display = result.skinToneLowConfidence ? "block" : "none";
  }

  // 4. Render Confidence
  if (result.confidence !== null && result.confidence !== undefined) {
    DOM.confidence.textContent = `${Math.round(result.confidence)}% match`;
    DOM.confidence.style.display = "inline-block";
  } else {
    DOM.confidence.style.display = "none";
  }

  // 5. Render Recommended Glasses (allGlasses first: the default try-on frame is looked up in it)
  state.allGlasses = result.allGlasses || [];
  renderGlasses(result.glasses || []);

  // 6. Render Hairstyle Suggestions
  renderHairstyle(result.hairstyle || []);

  // 7. Render Grooming Tips
  renderGrooming(result.grooming || []);

  // 8. Render Outfit Colors (wear + avoid)
  const outfitColors = result.outfitColors || { wear: [], avoid: [] };
  renderOutfitColors(outfitColors.wear || [], outfitColors.avoid || []);
}

/**
 * Render Recommended Glasses Cards
 */
function renderGlasses(glassesList) {
  DOM.glassesGrid.innerHTML = "";

  if (!glassesList || glassesList.length === 0) {
    DOM.glassesGrid.innerHTML = `<p class="outfit-empty">No glasses recommendations available for this analysis.</p>`;
    return;
  }

  glassesList.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "glasses-card";
    if (index === 0) card.classList.add("selected");
    card.setAttribute("tabindex", "0");
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Select ${item.name}`);

    card.innerHTML = `
      <div>
        <div class="glasses-img-box">
          <img src="${item.image || ""}" alt="${item.name}" onerror="this.style.visibility='hidden';">
        </div>
        <div class="glasses-name">${item.name}</div>
        ${(item.ids || []).length > 1 ? `<div class="glasses-count">${item.ids.length} matching frames</div>` : ""}
        <div class="glasses-reason">${item.reason}</div>
      </div>
      <button type="button" class="try-btn-sm">${index === 0 ? "Selected" : "Try On"}</button>
    `;

    card.addEventListener("click", () => selectGlasses(card, item));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectGlasses(card, item);
      }
    });

    DOM.glassesGrid.appendChild(card);
  });

  // Select first glasses style by default for try-on preview
  if (glassesList.length > 0) {
    updateTryOnPreview(glassesList[0]);
  } else {
    showFrame(0);
  }
}

function selectGlasses(cardEl, glassesObj) {
  const allCards = DOM.glassesGrid.querySelectorAll(".glasses-card");
  allCards.forEach((c) => {
    c.classList.remove("selected");
    const btn = c.querySelector(".try-btn-sm");
    if (btn) btn.textContent = "Try On";
  });

  cardEl.classList.add("selected");
  const selectedBtn = cardEl.querySelector(".try-btn-sm");
  if (selectedBtn) selectedBtn.textContent = "Selected";

  state.selectedGlasses = glassesObj;
  updateTryOnPreview(glassesObj);
}

// A recommended card was picked: put its first matching real frame on the face.
function updateTryOnPreview(glassesObj) {
  if (!glassesObj) return;
  const firstId = (glassesObj.ids || [])[0];
  const index = state.allGlasses.findIndex((g) => g.id === firstId);
  showFrame(index >= 0 ? index : 0, glassesObj.name, glassesObj.reason);
}

// Previous / Next: cycle through EVERY frame, recommended or not.
function stepGlasses(step) {
  const count = state.allGlasses.length;
  if (count === 0) return;
  const index = (state.tryonIndex + step + count) % count;
  const frame = state.allGlasses[index];
  const recs = (state.analysisResult && state.analysisResult.glasses) || [];
  const rec = recs.find((r) => (r.ids || []).includes(frame.id));
  markSelectedCard(rec ? recs.indexOf(rec) : -1);
  if (rec) {
    showFrame(index, rec.name, `Recommended for your face shape: ${rec.reason}`);
  } else {
    const label = frame.styles.length ? frame.styles.join(" / ") : "Other";
    showFrame(index, `${label.charAt(0).toUpperCase()}${label.slice(1)} frame`,
      "Not one of your top recommendations, but feel free to try it on.");
  }
}

function showFrame(index, title, reason) {
  const count = state.allGlasses.length;
  DOM.tryonCounter.textContent = count ? `${index + 1} / ${count}` : "0 / 0";
  if (count === 0) {
    DOM.tryonGlassesOverlay.hidden = true;
    return;
  }
  state.tryonIndex = index;
  DOM.tryonGlassesOverlay.src = state.allGlasses[index].image;
  DOM.tryonGlassesOverlay.hidden = false;
  if (title) DOM.tryonGlassesTitle.textContent = title;
  if (reason) DOM.tryonGlassesReason.textContent = reason;
  updateGlassesTransform();
}

// Highlights the recommended card at cardIndex (-1 = none).
function markSelectedCard(cardIndex) {
  DOM.glassesGrid.querySelectorAll(".glasses-card").forEach((c, i) => {
    c.classList.toggle("selected", i === cardIndex);
    const btn = c.querySelector(".try-btn-sm");
    if (btn) btn.textContent = i === cardIndex ? "Selected" : "Try On";
  });
}

function updateGlassesTransform() {
  if (DOM.tryonGlassesOverlay) {
    DOM.tryonGlassesOverlay.style.left = `${state.tryonLeft}%`;
    DOM.tryonGlassesOverlay.style.top = `${state.tryonTop}%`;
    DOM.tryonGlassesOverlay.style.width = `${state.tryonScale}%`;
  }
}

/**
 * Renders one grid of color swatch cards (shared by the "wear" and
 * "avoid" rows) from backend color objects: {name, hex}.
 */
function renderColorSwatchGrid(gridEl, colors, emptyMessage) {
  gridEl.innerHTML = "";

  if (!colors || colors.length === 0) {
    gridEl.innerHTML = `<p class="outfit-empty">${emptyMessage}</p>`;
    return;
  }

  colors.forEach((swatch) => {
    const swatchCard = document.createElement("div");
    swatchCard.className = "color-swatch-card";

    swatchCard.innerHTML = `
      <div class="color-block" style="background-color: ${swatch.hex};"></div>
      <div class="color-details">
        <span class="color-name">${swatch.name}</span>
        <span class="color-hex">${swatch.hex}</span>
      </div>
    `;

    gridEl.appendChild(swatchCard);
  });
}

/**
 * Render Outfit Colors (wear + avoid), from recommendations.py via the backend.
 */
function renderOutfitColors(wearColors, avoidColors) {
  renderColorSwatchGrid(DOM.colorSwatchesGrid, wearColors, "No color recommendations available yet.");
  renderColorSwatchGrid(DOM.avoidColorsGrid, avoidColors, "No colors to avoid — nothing flagged for this skin tone.");
}

/**
 * Render Hairstyle Suggestions, from recommendations.py via the backend.
 */
function renderHairstyle(hairstyleList) {
  DOM.hairstyleList.innerHTML = "";

  if (!hairstyleList || hairstyleList.length === 0) {
    DOM.hairstyleList.innerHTML = `<p class="outfit-empty">No hairstyle recommendations available for this analysis.</p>`;
    return;
  }

  hairstyleList.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "outfit-card";

    const tag = item.less_popular ? '<span class="less-popular-tag">less popular</span>' : '';
    card.innerHTML = `
      <div class="outfit-number">${idx + 1}</div>
      <div>
        <div class="outfit-text">${item.name}${tag}</div>
        <div class="outfit-text-why">${item.why}</div>
      </div>
    `;

    DOM.hairstyleList.appendChild(card);
  });
}

/**
 * Render Grooming Tips, from recommendations.py via the backend.
 */
function renderGrooming(tips) {
  DOM.groomingList.innerHTML = "";

  if (!tips || tips.length === 0) {
    DOM.groomingList.innerHTML = `<p class="outfit-empty">No grooming tips available for this analysis.</p>`;
    return;
  }

  tips.forEach((tip) => {
    const p = document.createElement("p");
    p.className = "section-description";
    p.style.marginTop = "0";
    p.textContent = `• ${tip}`;
    DOM.groomingList.appendChild(p);
  });
}

// ==========================================================================
// 12. APPLICATION RESET (TRY ANOTHER PHOTO)
// ==========================================================================
function resetApplication() {
  // Revoke object URL to prevent memory leaks
  if (state.photoUrl) {
    URL.revokeObjectURL(state.photoUrl);
  }

  // Reset state object
  state.photoFile = null;
  state.photoUrl = null;
  state.gender = null;
  state.occasion = null;
  state.isAnalyzing = false;
  state.analysisResult = null;
  state.selectedGlasses = null;
  state.allGlasses = [];
  state.tryonIndex = 0;
  if (DOM.tryonGlassesOverlay) DOM.tryonGlassesOverlay.hidden = true;
  state.tryonLeft = 50;
  state.tryonTop = 33;
  state.tryonScale = 42;

  syncSlidersUI();

  // Reset form inputs & UI
  if (DOM.photoInput) DOM.photoInput.value = "";
  if (DOM.photoPreview) DOM.photoPreview.src = "";
  if (DOM.previewContainer) DOM.previewContainer.classList.remove("active");
  if (DOM.uploadDropzone) DOM.uploadDropzone.style.display = "flex";

  // Reset Gender options
  if (DOM.genderOptions) {
    const genderBtns = DOM.genderOptions.querySelectorAll("[data-gender]");
    genderBtns.forEach((btn) => {
      btn.classList.remove("selected");
      btn.setAttribute("aria-checked", "false");
    });
  }

  // Reset Occasion options
  if (DOM.occasionOptions) {
    const occasionBtns = DOM.occasionOptions.querySelectorAll("[data-occasion]");
    occasionBtns.forEach((btn) => {
      btn.classList.remove("selected");
      btn.setAttribute("aria-checked", "false");
    });
  }

  clearError();
  showUploadScreen();
}
