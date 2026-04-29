const state = {
  slides: [],
  availableExperiments: [],
  filterOptions: {
    diagnosis: [],
    infiltration: [],
  },
  activeFilters: {
    diagnosis: "",
    infiltration: "",
  },
  manifest: null,
  cells: null,
  viewer: null,
  currentSlideKey: null,
  currentExperiment: null,
  pendingExperiment: null,
  topbarStatusHideTimer: null,
  topbarStatus: "idle",
  pendingViewportState: null,
  redrawPending: false,
  sidebarRedrawPending: false,
  sidebarRedrawTimer: null,
  viewportQuerySyncTimer: null,
  viewportApplyTimer: null,
  slideLoadToken: 0,
  dotDiameterImagePx: null,
  pendingDotRadiusValue: null,
  boxSizeImagePx: null,
  pendingBoxSizeValue: null,
  pendingClusterFilterIds: null,
  partitionFillAlpha: 0.25,
  binaryDotThreshold: 0.5,
  activeClusterFilters: new Set(),
  topbarSelectionRevealed: false,
  activeImageLayer: "srhvhe",
  pendingImageLayerFallback: null,
  layoutResizeActive: false,
  sidebarCollapsed: false,
};

const TEXT_RENDER_LIMIT = 2200;
const SCORE_LABEL_VIEW_WIDTH_RATIO = 0.34;
const CLUSTER_LABEL_VIEW_WIDTH_RATIO = 0.28;
const CONTRIBUTION_LABEL_VIEW_WIDTH_RATIO = 0.24;
const FILTER_ALL_VALUE = "__all__";
const CELL_BOX_SIZE_PX = 48;
const MIN_SCREEN_BOX_SIZE_PX = 8;
const DOT_DIAMETER_IMAGE_MIN = 2;
const DOT_DIAMETER_IMAGE_MAX = 64;
const DEFAULT_DOT_DIAMETER_IMAGE_PX = 32;
const BOX_SIZE_IMAGE_MIN = 2;
const BOX_SIZE_IMAGE_MAX = 64;
const DEFAULT_BOX_SIZE_IMAGE_PX = CELL_BOX_SIZE_PX;
const DEFAULT_PARTITION_FILL_ALPHA = 0.25;
const PARTITION_SAMPLE_STEP_PX = 8;
const PARTITION_ZOOMED_OUT_SAMPLE_STEP_PX = 4;
const PARTITION_HIGH_RES_VIEW_WIDTH_RATIO = 0.2;
const PARTITION_BUCKET_SIZE_PX = 56;
const PARTITION_VIEW_MARGIN_PX = PARTITION_SAMPLE_STEP_PX * 6;
const SIDEBAR_REDRAW_DELAY_MS = 90;
const OVERLAY_QUERY_PARAM = "overlays";
const EXPERIMENT_QUERY_PARAM = "experiment";
const LEGACY_EXPERIMENT_QUERY_PARAM = "k";
const DOT_SIZE_QUERY_PARAM = "dot";
const BOX_SIZE_QUERY_PARAM = "box";
const PARTITION_ALPHA_QUERY_PARAM = "alpha";
const BINARY_THRESHOLD_QUERY_PARAM = "threshold";
const VIEW_QUERY_PARAM = "view";
const VIEW_MODE_CONTINUOUS = "continuous";
const VIEW_MODE_BINARY = "binary";
const CLUSTER_QUERY_PARAM = "clusters";
const IMAGE_LAYER_QUERY_PARAM = "layer";
const VIEWPORT_X_QUERY_PARAM = "vx";
const VIEWPORT_Y_QUERY_PARAM = "vy";
const VIEWPORT_ZOOM_QUERY_PARAM = "vz";
const DEFAULT_VIEWPORT_CENTER_X = 0.5;
const DEFAULT_VIEWPORT_CENTER_Y = 0.5;
const DEFAULT_VIEWPORT_HEIGHT_RATIO = 1.0;
const VIEWPORT_AUTO_APPLY_DELAY_MS = 220;
const SIDEBAR_WIDTH_MIN_PERCENT = 18;
const SIDEBAR_WIDTH_MAX_PERCENT = 55;
const SIDEBAR_WIDTH_MIN_PX = 280;
const SLIDE_CELLS_PATH = "cells.json";
const IMAGE_LAYER_TILE_SOURCES = {
  srhvhe: "srhvhe.dzi",
  srhrgb: "srhrgb.dzi",
};
const OVERLAY_BOX_STROKE_STYLE = "#ffff00";
const OVERLAY_TEXT_STROKE_STYLE = "rgb(250 250 249 / 0.92)";
const OVERLAY_TEXT_FILL_STYLE = "rgb(41 37 36)";
const SCORE_HISTOGRAM_FILL_STYLE = "rgb(59 130 246)";
const CLUSTER_HISTOGRAM_FILL_STYLE = "rgb(87 83 78)";
const CHART_BACKGROUND_STYLE = "rgb(250 250 249)";
const CHART_GRID_STROKE_STYLE = "rgb(87 83 78 / 0.12)";
const CHART_AXIS_STROKE_STYLE = "rgb(87 83 78 / 0.28)";
const CHART_TEXT_FILL_STYLE = "rgb(120 113 108)";
const TUMOR_SCORE_COLOR_STOPS = [
  { position: 0, color: [26, 152, 80] },
  { position: 0.32, color: [217, 239, 139] },
  { position: 0.58, color: [253, 219, 199] },
  { position: 0.78, color: [239, 138, 98] },
  { position: 1, color: [178, 24, 43] },
];
const DEFAULT_BINARY_DOT_SCORE_THRESHOLD = 0.5;
const BINARY_DOT_STROKE_STYLE = "rgb(41 37 36 / 0.42)";
const INFILTRATION_LABELS = {
  "0": "Normal (0)",
  "1": "Atypical Cells (1)",
  "2": "Sparse Tumor (2)",
  "3": "Dense Tumor (3)",
  UNK: "UNK",
};

document.addEventListener("DOMContentLoaded", () => {
  void bootstrapPortal().catch((error) => {
    console.error(error);
    setTopbarStatus("error");
    setViewportEditorStatus(String(error));
  });
});

async function bootstrapPortal() {
  setTopbarStatus("loading");
  applyUiStateFromQuery();
  const slidesPayload = await fetchJson(window.SILICA_PORTAL.slidesUrl);
  state.slides = normalizeSlidesPayload(slidesPayload.slides);
  state.availableExperiments = normalizeExperimentNames(slidesPayload.experiments);
  state.filterOptions = resolveFilterOptions(slidesPayload.filters, state.slides);
  if (state.slides.length === 0) {
    throw new Error("No slides were returned by /api/slides");
  }
  const initialSlideKey = resolveInitialSlideKey(slidesPayload.default_slide_key);
  state.currentExperiment = resolveInitialExperiment(
    slidesPayload.default_experiment,
    initialSlideKey,
  );
  state.pendingExperiment = null;
  populateFilterSelectors();
  populateSlideSelector();
  populateExperimentSelector(initialSlideKey);
  bindControls();
  await changeSlide(initialSlideKey, {
    syncEmptyFilters: true,
  });
  initializeViewer();
}

function revealTopbarSelectionLabels() {
  if (state.topbarSelectionRevealed) {
    return;
  }
  state.topbarSelectionRevealed = true;
  document.body.classList.add("topbar-selection-revealed");
}

function bindTopbarSelectionReveal() {
  const revealOnce = () => {
    revealTopbarSelectionLabels();
  };
  const topbarFilterPill = document.querySelector(".topbar-filter-pill");

  topbarFilterPill?.addEventListener("pointerenter", revealOnce, {
    once: true,
    passive: true,
  });

  for (const eventName of ["pointerdown", "keydown", "wheel", "touchstart"]) {
    document.addEventListener(eventName, revealOnce, {
      once: true,
      passive: true,
    });
  }
}

function normalizeSlidesPayload(rawSlides) {
  if (!Array.isArray(rawSlides)) {
    return [];
  }

  return rawSlides.map((slide) => ({
    ...slide,
    available_experiments: normalizeExperimentNames(slide.available_experiments),
    diagnosis: normalizeFilterValue(slide.diagnosis),
    infiltration: normalizeFilterValue(slide.infiltration),
  }));
}

function normalizeExperimentNames(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  return sortFilterValues(
    values
      .map((value) => String(value ?? "").trim())
      .filter(Boolean),
  );
}

function resolveFilterOptions(rawFilters, slides) {
  const derivedDiagnosis = collectUniqueValues(slides, "diagnosis");
  const derivedInfiltration = collectUniqueValues(slides, "infiltration");

  return {
    diagnosis: Array.isArray(rawFilters?.diagnosis)
      ? sortFilterValues(rawFilters.diagnosis.map(normalizeFilterValue))
      : derivedDiagnosis,
    infiltration: Array.isArray(rawFilters?.infiltration)
      ? sortFilterValues(rawFilters.infiltration.map(normalizeFilterValue))
      : derivedInfiltration,
  };
}

function collectUniqueValues(slides, key) {
  return sortFilterValues(
    Array.from(new Set(slides.map((slide) => normalizeFilterValue(slide[key])))),
  );
}

function sortFilterValues(values) {
  return [...new Set(values)].sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" }),
  );
}

function normalizeFilterValue(value) {
  if (value === null || value === undefined) {
    return "UNK";
  }
  const normalized = String(value).trim();
  return normalized || "UNK";
}

function getFilterDisplayLabel(filterKey, value, defaultLabel) {
  if (filterKey === "infiltration") {
    return INFILTRATION_LABELS[value] || value || defaultLabel;
  }
  return value || defaultLabel;
}

function populateSlideHeader() {
  document.getElementById("totalCellCount").textContent = formatInteger(
    state.cells.cell_count,
  );
  const slideStatistics = state.manifest?.slide_statistics || {};
  setSavedPercentMetric(
    "softSlideTumorScore",
    slideStatistics.soft_slide_tumor_probability,
  );
  setSavedPercentMetric(
    "hardSlideTumorScore",
    slideStatistics.hard_slide_tumor_probability,
  );
  setSavedPercentMetric(
    "areaSlideTumorScore",
    slideStatistics.area_soft_slide_tumor_probability,
  );
}

function setPercentMetric(elementId, value) {
  const element = document.getElementById(elementId);
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    element.textContent = "-";
    element.style.color = "";
    return;
  }

  element.textContent = `${(numericValue * 100).toFixed(1)}%`;
  element.style.color = getTumorScoreColor(numericValue);
}

function setSavedPercentMetric(elementId, value) {
  const element = document.getElementById(elementId);
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    element.textContent = "-";
    element.style.color = "";
    return;
  }

  element.textContent = `${(numericValue * 100).toFixed(1)}%`;
  element.style.color = getTumorScoreColor(numericValue);
}

function meanTumorScoreForIndices(indices) {
  if (!indices.length) {
    return null;
  }

  let tumorScoreSum = 0;
  for (const index of indices) {
    tumorScoreSum += state.cells.tumor_score[index];
  }
  return tumorScoreSum / indices.length;
}

function getClusterCounts(indices = null) {
  const counts = new Map();
  const sourceIndices =
    indices ?? Array.from({ length: state.cells.dominant_cluster.length }, (_, index) => index);
  for (const index of sourceIndices) {
    const cluster = state.cells.dominant_cluster[index];
    counts.set(cluster, (counts.get(cluster) || 0) + 1);
  }
  return counts;
}

function getSortedClustersByCount() {
  return Array.from(getClusterCounts().entries()).sort((left, right) => {
    if (right[1] !== left[1]) {
      return right[1] - left[1];
    }
    return left[0] - right[0];
  });
}

function getAllClusterIds() {
  return getSortedClustersByCount().map(([cluster]) => cluster);
}

function formatClusterFilterValue(clusterIds) {
  return [...clusterIds].sort((left, right) => left - right).join(",");
}

function setClusterFilterStatus(message) {
  const status = document.getElementById("clusterFilterStatus");
  if (!status) {
    return;
  }
  status.textContent = message || "";
}

function syncClusterFilterEditor() {
  const input = document.getElementById("clusterFilterInput");
  if (!input) {
    return;
  }

  if (!state.cells) {
    if (state.pendingClusterFilterIds === "none") {
      input.value = "none";
      setClusterFilterStatus("Showing no clusters.");
      return;
    }
    if (Array.isArray(state.pendingClusterFilterIds)) {
      input.value = formatClusterFilterValue(state.pendingClusterFilterIds);
      setClusterFilterStatus(
        `Pending ${formatInteger(state.pendingClusterFilterIds.length)} cluster filter(s).`,
      );
      return;
    }
    input.value = "";
    setClusterFilterStatus("Leave empty for all clusters.");
    return;
  }

  const allClusterIds = getAllClusterIds();
  const selectedClusterIds = Array.from(state.activeClusterFilters).sort((left, right) => left - right);
  if (selectedClusterIds.length === 0) {
    input.value = "none";
    setClusterFilterStatus("Showing no clusters.");
    return;
  }
  if (
    allClusterIds.length > 0 &&
    allClusterIds.every((clusterId) => state.activeClusterFilters.has(clusterId))
  ) {
    input.value = "";
    setClusterFilterStatus(`Showing all ${formatInteger(allClusterIds.length)} clusters.`);
    return;
  }

  input.value = formatClusterFilterValue(selectedClusterIds);
  setClusterFilterStatus(
    `Showing ${formatInteger(selectedClusterIds.length)} of ${formatInteger(
      allClusterIds.length,
    )} clusters.`,
  );
}

function applyClusterFilterInputValue() {
  if (!state.cells) {
    return;
  }
  const input = document.getElementById("clusterFilterInput");
  if (!input) {
    return;
  }

  const allClusterIds = getAllClusterIds();
  const rawValue = input.value.trim();
  const normalizedValue = rawValue.toLowerCase();
  if (!rawValue || normalizedValue === "all") {
    state.activeClusterFilters = new Set(allClusterIds);
    syncClusterFilterEditor();
    syncUiStateQuery();
    scheduleRedraw();
    scheduleSidebarRedraw({ immediate: true });
    return;
  }
  if (normalizedValue === "none") {
    state.activeClusterFilters = new Set();
    syncClusterFilterEditor();
    syncUiStateQuery();
    scheduleRedraw();
    scheduleSidebarRedraw({ immediate: true });
    return;
  }

  const requestedClusterIds = rawValue
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => Number.parseInt(value, 10))
    .filter((value) => Number.isInteger(value));
  if (requestedClusterIds.length === 0) {
    state.activeClusterFilters = new Set(allClusterIds);
    syncClusterFilterEditor();
    setClusterFilterStatus("No valid cluster ids in list. Showing all clusters.");
    syncUiStateQuery();
    scheduleRedraw();
    scheduleSidebarRedraw({ immediate: true });
    return;
  }
  const validClusterIds = [...new Set(requestedClusterIds)].filter((clusterId) =>
    allClusterIds.includes(clusterId),
  );

  state.activeClusterFilters =
    validClusterIds.length > 0 ? new Set(validClusterIds) : new Set(allClusterIds);
  syncClusterFilterEditor();
  if (validClusterIds.length === 0) {
    setClusterFilterStatus("No matching cluster ids in list. Showing all clusters.");
  }
  syncUiStateQuery();
  scheduleRedraw();
  scheduleSidebarRedraw({ immediate: true });
}

function populateSlideSelector() {
  const matchingSlides = getFilteredSlides();
  const slideSelect = document.getElementById("slideSelect");
  const previousValue = slideSelect.value;
  slideSelect.replaceChildren();

  if (matchingSlides.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No slides";
    slideSelect.appendChild(option);
    slideSelect.disabled = true;
    syncSlideNavigationButtons();
    return;
  }

  for (const slide of matchingSlides) {
    const option = document.createElement("option");
    option.value = slide.key;
    option.textContent = slide.label;
    slideSelect.appendChild(option);
  }

  slideSelect.disabled = false;
  if (matchingSlides.some((slide) => slide.key === previousValue)) {
    slideSelect.value = previousValue;
    syncSlideNavigationButtons();
    return;
  }
  slideSelect.value = matchingSlides[0].key;
  syncSlideNavigationButtons();
}

function syncSlideNavigationButtons() {
  const previousButton = document.getElementById("previousSlideButton");
  const nextButton = document.getElementById("nextSlideButton");
  if (!previousButton || !nextButton) {
    return;
  }

  const matchingSlides = getFilteredSlides();
  const currentIndex = matchingSlides.findIndex(
    (slide) => slide.key === state.currentSlideKey,
  );
  previousButton.disabled = currentIndex <= 0;
  nextButton.disabled =
    currentIndex === -1 || currentIndex >= matchingSlides.length - 1;
}

function populateFilterSelectors() {
  populateFilterSelector("diagnosisSelect", "diagnosis", "Diagnosis");
  populateFilterSelector("infiltrationSelect", "infiltration", "Infiltration");
}

function populateFilterSelector(selectId, filterKey, defaultLabel) {
  const select = document.getElementById(selectId);
  select.replaceChildren();

  const placeholderOption = document.createElement("option");
  placeholderOption.value = "";
  placeholderOption.textContent = defaultLabel;
  placeholderOption.hidden = true;
  select.appendChild(placeholderOption);

  const defaultOption = document.createElement("option");
  defaultOption.value = FILTER_ALL_VALUE;
  defaultOption.textContent = "All";
  select.appendChild(defaultOption);

  for (const value of state.filterOptions[filterKey]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = getFilterDisplayLabel(filterKey, value, defaultLabel);
    if (willFilterOptionChangeContext(filterKey, value)) {
      option.classList.add("will-change-slide");
    }
    select.appendChild(option);
  }

  select.value = state.activeFilters[filterKey] || "";
}

function willFilterOptionChangeContext(filterKey, value) {
  const oppositeFilterKey =
    filterKey === "diagnosis" ? "infiltration" : "diagnosis";
  const oppositeFilterValue = state.activeFilters[oppositeFilterKey];
  if (!oppositeFilterValue) {
    return false;
  }

  return !state.slides.some(
    (slide) =>
      slide[filterKey] === value &&
      slide[oppositeFilterKey] === oppositeFilterValue,
  );
}

function resolveInitialSlideKey(defaultSlideKey) {
  const searchParams = new URLSearchParams(window.location.search);
  const requestedSlideKey = searchParams.get("slide");
  if (requestedSlideKey && getSlideEntry(requestedSlideKey)) {
    return requestedSlideKey;
  }
  if (defaultSlideKey && getSlideEntry(defaultSlideKey)) {
    return defaultSlideKey;
  }
  return state.slides[0]?.key ?? null;
}

function getSlideEntry(slideKey) {
  return state.slides.find((slide) => slide.key === slideKey) || null;
}

function getSlideAvailableExperiments(slideKey) {
  return getSlideEntry(slideKey)?.available_experiments ?? [];
}

function resolveInitialExperiment(defaultExperiment, slideKey) {
  const slideAvailableExperiments = getSlideAvailableExperiments(slideKey);
  const candidateExperiments = [
    state.pendingExperiment,
    typeof defaultExperiment === "string" ? defaultExperiment.trim() : "",
    slideAvailableExperiments[0],
    state.availableExperiments[0],
  ].filter(Boolean);

  for (const experimentName of candidateExperiments) {
    if (
      slideAvailableExperiments.length === 0 ||
      slideAvailableExperiments.includes(experimentName)
    ) {
      return experimentName;
    }
  }
  return null;
}

function ensureValidCurrentExperiment(slideKey = state.currentSlideKey) {
  const slideAvailableExperiments = getSlideAvailableExperiments(slideKey);
  if (slideAvailableExperiments.length === 0) {
    state.currentExperiment = null;
    return null;
  }
  if (slideAvailableExperiments.includes(state.currentExperiment)) {
    return state.currentExperiment;
  }
  const fallbackExperiment = slideAvailableExperiments[0];
  state.currentExperiment = fallbackExperiment;
  return fallbackExperiment;
}

function populateExperimentSelector(slideKey = state.currentSlideKey) {
  const select = document.getElementById("experimentSelect");
  if (!select) {
    return;
  }

  const slideAvailableExperiments = getSlideAvailableExperiments(slideKey);
  const resolvedExperiment = ensureValidCurrentExperiment(slideKey);
  select.replaceChildren();

  if (state.availableExperiments.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Experiment";
    select.appendChild(option);
    select.disabled = true;
    return;
  }

  for (const experimentName of state.availableExperiments) {
    const option = document.createElement("option");
    option.value = experimentName;
    option.textContent = experimentName;
    option.disabled = !slideAvailableExperiments.includes(experimentName);
    select.appendChild(option);
  }

  select.disabled = slideAvailableExperiments.length === 0;
  if (resolvedExperiment !== null) {
    select.value = resolvedExperiment;
  }
}

function getFilteredSlidesFor(filters) {
  return state.slides.filter((slide) => {
    if (filters.diagnosis && slide.diagnosis !== filters.diagnosis) {
      return false;
    }
    if (filters.infiltration && slide.infiltration !== filters.infiltration) {
      return false;
    }
    return true;
  });
}

function getFilteredSlides() {
  return getFilteredSlidesFor(state.activeFilters);
}

function resolveFilterTransition(nextFilters) {
  const requestedFilters = {
    ...state.activeFilters,
    ...nextFilters,
  };
  const strictMatches = getFilteredSlidesFor(requestedFilters);
  if (strictMatches.length > 0) {
    return {
      filters: requestedFilters,
      matchingSlides: strictMatches,
    };
  }

  const changedKeys = Object.keys(nextFilters);
  if (changedKeys.length !== 1) {
    return {
      filters: requestedFilters,
      matchingSlides: strictMatches,
    };
  }

  const changedKey = changedKeys[0];
  const changedValue = requestedFilters[changedKey];
  if (!changedValue) {
    return {
      filters: requestedFilters,
      matchingSlides: strictMatches,
    };
  }

  const categoryMatches = state.slides.filter((slide) => slide[changedKey] === changedValue);
  if (categoryMatches.length === 0) {
    return {
      filters: requestedFilters,
      matchingSlides: strictMatches,
    };
  }

  const firstMatch = categoryMatches[0];
  const adjustedFilters = {
    diagnosis:
      changedKey === "diagnosis" ? changedValue : firstMatch.diagnosis,
    infiltration:
      changedKey === "infiltration" ? changedValue : firstMatch.infiltration,
  };
  return {
    filters: adjustedFilters,
    matchingSlides: getFilteredSlidesFor(adjustedFilters),
  };
}

function bindControls() {
  bindTopbarSelectionReveal();
  bindSidebarCollapse();

  document.getElementById("diagnosisSelect").addEventListener("change", (event) => {
    revealTopbarSelectionLabels();
    applyFilters({
      diagnosis:
        event.target.value === FILTER_ALL_VALUE ? "" : event.target.value,
    });
  });

  document
    .getElementById("infiltrationSelect")
    .addEventListener("change", (event) => {
      revealTopbarSelectionLabels();
      applyFilters({
        infiltration:
          event.target.value === FILTER_ALL_VALUE ? "" : event.target.value,
      });
    });

  document.getElementById("slideSelect").addEventListener("change", (event) => {
    revealTopbarSelectionLabels();
    if (event.target.value) {
      syncUiStateQuery(event.target.value);
      void changeSlide(event.target.value, {
        syncEmptyFilters: true,
        preserveViewport: true,
      });
    }
  });
  document.getElementById("previousSlideButton").addEventListener("click", () => {
    navigateFilteredSlides(-1);
  });
  document.getElementById("nextSlideButton").addEventListener("click", () => {
    navigateFilteredSlides(1);
  });
  document.getElementById("experimentSelect").addEventListener("change", (event) => {
    const requestedExperiment = String(event.target.value ?? "").trim();
    if (!requestedExperiment) {
      return;
    }
    if (!getSlideAvailableExperiments(state.currentSlideKey).includes(requestedExperiment)) {
      populateExperimentSelector(state.currentSlideKey);
      return;
    }
    state.currentExperiment = requestedExperiment;
    syncUiStateQuery();
    if (state.currentSlideKey) {
      void changeSlide(state.currentSlideKey);
    }
  });
  document.querySelector(".image-layer-pill").addEventListener("click", (event) => {
    const button = event.target.closest(".image-layer-button");
    if (!button) {
      return;
    }
    setActiveImageLayer(button.dataset.imageLayer);
  });
  bindLayoutResizeHandle();
  const clusterFilterInput = document.getElementById("clusterFilterInput");
  clusterFilterInput.addEventListener("change", () => {
    applyClusterFilterInputValue();
  });
  clusterFilterInput.addEventListener("blur", () => {
    applyClusterFilterInputValue();
  });
  clusterFilterInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    applyClusterFilterInputValue();
  });

  for (const input of document.querySelectorAll("input[type='checkbox']")) {
    input.addEventListener("change", () => {
      syncUiStateQuery();
      scheduleRedraw();
    });
  }
  document.getElementById("toggleDots").addEventListener("change", () => {
    syncOverlayControlVisibility();
  });
  document.getElementById("toggleBoxes").addEventListener("change", () => {
    syncOverlayControlVisibility();
  });
  document.getElementById("togglePartitionFill").addEventListener("change", () => {
    syncOverlayControlVisibility();
  });
  document.getElementById("toggleBinaryDots").addEventListener("change", () => {
    syncOverlayControlVisibility();
  });
  document.getElementById("dotSizeSlider").addEventListener("input", (event) => {
    const sliderValue = Number(event.target.value);
    if (!Number.isFinite(sliderValue)) {
      return;
    }
    const clampedSliderValue = clampNumber(
      sliderValue,
      DOT_DIAMETER_IMAGE_MIN,
      DOT_DIAMETER_IMAGE_MAX,
    );
    state.dotDiameterImagePx = clampedSliderValue;
    state.pendingDotRadiusValue = null;
    updateDotSizeControl(clampedSliderValue);
    syncUiStateQuery();
    scheduleRedraw();
  });
  document.getElementById("boxSizeSlider").addEventListener("input", (event) => {
    const sliderValue = Number(event.target.value);
    if (!Number.isFinite(sliderValue)) {
      return;
    }
    const clampedSliderValue = clampNumber(
      sliderValue,
      BOX_SIZE_IMAGE_MIN,
      BOX_SIZE_IMAGE_MAX,
    );
    state.boxSizeImagePx = clampedSliderValue;
    state.pendingBoxSizeValue = null;
    updateBoxSizeControl(clampedSliderValue);
    syncUiStateQuery();
    scheduleRedraw();
  });
  document.getElementById("partitionAlphaSlider").addEventListener("input", (event) => {
    const sliderValue = Number(event.target.value);
    if (!Number.isFinite(sliderValue)) {
      return;
    }
    state.partitionFillAlpha = clampNumber(sliderValue / 100, 0, 1);
    updatePartitionAlphaControl(state.partitionFillAlpha);
    syncUiStateQuery();
    scheduleRedraw();
  });
  document.getElementById("binaryThresholdSlider").addEventListener("input", (event) => {
    const sliderValue = Number(event.target.value);
    if (!Number.isFinite(sliderValue)) {
      return;
    }
    state.binaryDotThreshold = clampNumber(sliderValue / 100, 0, 1);
    updateBinaryThresholdControl(state.binaryDotThreshold);
    syncUiStateQuery();
    scheduleRedraw();
  });
  const viewportForm = document.getElementById("viewportBoundsForm");
  viewportForm.addEventListener("submit", (event) => {
    event.preventDefault();
  });
  for (const inputId of ["viewportInputVx", "viewportInputVy", "viewportInputVz"]) {
    const input = document.getElementById(inputId);
    input.addEventListener("input", () => {
      scheduleViewportAutoApply();
    });
    input.addEventListener("change", () => {
      scheduleViewportAutoApply({ immediate: true });
    });
  }

  syncOverlayControlVisibility();
  syncClusterFilterEditor();
  updateDotSizeControl(state.pendingDotRadiusValue ?? DEFAULT_DOT_DIAMETER_IMAGE_PX);
  updateBoxSizeControl(state.pendingBoxSizeValue ?? DEFAULT_BOX_SIZE_IMAGE_PX);
  updatePartitionAlphaControl(state.partitionFillAlpha);
  updateBinaryThresholdControl(state.binaryDotThreshold);
  window.addEventListener("resize", scheduleRedraw);
  window.addEventListener("resize", () => {
    updateViewerViewportMargins();
    scheduleSidebarRedraw();
  });
}

function bindSidebarCollapse() {
  const buttons = [
    document.getElementById("sidebarCollapseButton"),
    document.getElementById("sidebarHeaderCollapseButton"),
  ].filter(Boolean);
  if (buttons.length === 0) {
    return;
  }
  for (const button of buttons) {
    button.addEventListener("click", () => {
      setSidebarCollapsed(!state.sidebarCollapsed);
    });
  }
  setSidebarCollapsed(state.sidebarCollapsed);
}

function bindLayoutResizeHandle() {
  const resizeHandle = document.getElementById("layoutResizeHandle");
  const layout = document.querySelector(".layout");
  if (!resizeHandle || !layout) {
    return;
  }

  resizeHandle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    state.layoutResizeActive = true;
    document.body.classList.add("is-resizing-layout");
    resizeHandle.setPointerCapture(event.pointerId);
    updateSidebarWidthFromPointer(event.clientX);
  });

  resizeHandle.addEventListener("pointermove", (event) => {
    if (!state.layoutResizeActive) {
      return;
    }
    updateSidebarWidthFromPointer(event.clientX);
  });

  const endResize = (event) => {
    if (!state.layoutResizeActive) {
      return;
    }
    state.layoutResizeActive = false;
    document.body.classList.remove("is-resizing-layout");
    if (resizeHandle.hasPointerCapture(event.pointerId)) {
      resizeHandle.releasePointerCapture(event.pointerId);
    }
    refreshViewerAfterLayoutResize();
  };

  resizeHandle.addEventListener("pointerup", endResize);
  resizeHandle.addEventListener("pointercancel", endResize);
}

function setSidebarCollapsed(isCollapsed) {
  state.sidebarCollapsed = isCollapsed;
  document.body.classList.toggle("sidebar-collapsed", isCollapsed);
  const buttons = [
    document.getElementById("sidebarCollapseButton"),
    document.getElementById("sidebarHeaderCollapseButton"),
  ].filter(Boolean);
  for (const button of buttons) {
    button.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
    button.setAttribute(
      "aria-label",
      isCollapsed ? "Expand sidebar" : "Collapse sidebar",
    );
  }
  refreshViewerAfterLayoutResize();
}

function navigateFilteredSlides(direction) {
  const matchingSlides = getFilteredSlides();
  const currentIndex = matchingSlides.findIndex(
    (slide) => slide.key === state.currentSlideKey,
  );
  if (currentIndex === -1) {
    syncSlideNavigationButtons();
    return;
  }

  const nextIndex = currentIndex + direction;
  if (nextIndex < 0 || nextIndex >= matchingSlides.length) {
    syncSlideNavigationButtons();
    return;
  }

  const nextSlideKey = matchingSlides[nextIndex].key;
  document.getElementById("slideSelect").value = nextSlideKey;
  syncSlideNavigationButtons();
  syncUiStateQuery(nextSlideKey);
  void changeSlide(nextSlideKey, { preserveViewport: true });
}

function updateSidebarWidthFromPointer(clientX) {
  const layout = document.querySelector(".layout");
  if (!layout) {
    return;
  }

  const layoutRect = layout.getBoundingClientRect();
  if (layoutRect.width <= 0) {
    return;
  }

  const minPercent = Math.max(
    SIDEBAR_WIDTH_MIN_PERCENT,
    (SIDEBAR_WIDTH_MIN_PX / layoutRect.width) * 100,
  );
  const rawPercent = ((clientX - layoutRect.left) / layoutRect.width) * 100;
  const sidebarWidthPercent = clampNumber(
    rawPercent,
    minPercent,
    SIDEBAR_WIDTH_MAX_PERCENT,
  );
  document.documentElement.style.setProperty(
    "--sidebar-width",
    `${sidebarWidthPercent.toFixed(2)}%`,
  );
  refreshViewerAfterLayoutResize();
}

function refreshViewerAfterLayoutResize() {
  if (state.viewer?.viewport) {
    if (
      typeof state.viewer.viewport.resize === "function" &&
      typeof state.viewer.viewport.getContainerSize === "function"
    ) {
      state.viewer.viewport.resize(state.viewer.viewport.getContainerSize(), true);
    }
    state.viewer.viewport.applyConstraints();
  }
  scheduleRedraw();
  scheduleSidebarRedraw();
  scheduleViewportQuerySync();
}

function applyFilters(nextFilters) {
  const { filters, matchingSlides } = resolveFilterTransition(nextFilters);
  state.activeFilters = filters;
  populateFilterSelectors();
  populateSlideSelector();
  if (matchingSlides.length === 0) {
    return;
  }

  const preferredSlideKey = matchingSlides.some(
    (slide) => slide.key === state.currentSlideKey,
  )
    ? state.currentSlideKey
    : matchingSlides[0].key;
  document.getElementById("slideSelect").value = preferredSlideKey;
  if (preferredSlideKey !== state.currentSlideKey) {
    void changeSlide(preferredSlideKey, { preserveViewport: true });
  }
}

function snapshotClusterFilterState() {
  if (state.cells) {
    const allClusterIds = getAllClusterIds();
    if (state.activeClusterFilters.size === 0) {
      return "none";
    }
    if (
      allClusterIds.length > 0 &&
      allClusterIds.every((clusterId) => state.activeClusterFilters.has(clusterId))
    ) {
      return "all";
    }
    return Array.from(state.activeClusterFilters).sort((left, right) => left - right);
  }
  if (state.pendingClusterFilterIds === "all" || state.pendingClusterFilterIds === "none") {
    return state.pendingClusterFilterIds;
  }
  if (Array.isArray(state.pendingClusterFilterIds)) {
    return [...state.pendingClusterFilterIds].sort((left, right) => left - right);
  }
  return null;
}

function restorePendingClusterFilterState(clusterFilterState) {
  if (
    clusterFilterState === "all" ||
    clusterFilterState === "none" ||
    Array.isArray(clusterFilterState)
  ) {
    state.pendingClusterFilterIds = clusterFilterState;
    return;
  }
  state.pendingClusterFilterIds = null;
}

async function changeSlide(slideKey, options = {}) {
  const { syncEmptyFilters = false, preserveViewport = false } = options;
  if (!slideKey) {
    throw new Error("No slide is available for the current filter selection");
  }
  const slideEntry = getSlideEntry(slideKey);
  if (!slideEntry) {
    throw new Error(`Unknown slide key: ${slideKey}`);
  }

  if (
    syncEmptyFilters &&
    (!state.activeFilters.diagnosis || !state.activeFilters.infiltration)
  ) {
    state.activeFilters = {
      ...state.activeFilters,
      diagnosis: state.activeFilters.diagnosis
        ? state.activeFilters.diagnosis
        : slideEntry.diagnosis,
      infiltration: state.activeFilters.infiltration
        ? state.activeFilters.infiltration
        : slideEntry.infiltration,
    };
    populateFilterSelectors();
    populateSlideSelector();
  }
  const resolvedExperiment = ensureValidCurrentExperiment(slideKey);
  if (resolvedExperiment === null) {
    throw new Error(`No experiments are available for slide ${slideKey}`);
  }
  const preserveCurrentViewerImage = Boolean(state.viewer) && state.currentSlideKey === slideKey;
  const preservedViewportState =
    preserveViewport && !preserveCurrentViewerImage ? captureViewportState() : null;
  const preservedClusterFilterState = snapshotClusterFilterState();

  const slideSelect = document.getElementById("slideSelect");
  const diagnosisSelect = document.getElementById("diagnosisSelect");
  const infiltrationSelect = document.getElementById("infiltrationSelect");
  const experimentSelect = document.getElementById("experimentSelect");
  const previousSlideButton = document.getElementById("previousSlideButton");
  const nextSlideButton = document.getElementById("nextSlideButton");
  const loadToken = state.slideLoadToken + 1;
  state.slideLoadToken = loadToken;
  setTopbarStatus("loading");
  slideSelect.disabled = true;
  diagnosisSelect.disabled = true;
  infiltrationSelect.disabled = true;
  experimentSelect.disabled = true;
  previousSlideButton.disabled = true;
  nextSlideButton.disabled = true;
  slideSelect.value = slideKey;
  populateExperimentSelector(slideKey);

  try {
    const manifest = await fetchJson(
      slidePortalAssetUrl(slideKey, resolvedExperiment, "slide_manifest.json"),
    );
    const cells = await fetchJson(
      slidePortalAssetUrl(slideKey, resolvedExperiment, SLIDE_CELLS_PATH),
    );
    if (loadToken !== state.slideLoadToken) {
      return;
    }

    state.currentSlideKey = slideKey;
    state.currentExperiment = resolvedExperiment;
    state.pendingExperiment = null;
    state.manifest = manifest;
    state.cells = cells;
    syncImageLayerControls();
    state.activeClusterFilters = new Set(getAllClusterIds());
    restorePendingClusterFilterState(preservedClusterFilterState);
    applyPendingClusterFilters();
    populateSlideHeader();
    populateExperimentSelector(slideKey);
    syncClusterFilterEditor();
    syncSlideQuery(slideKey);

    if (state.viewer) {
      if (preserveCurrentViewerImage) {
        setTopbarStatus("ready");
        scheduleRedraw();
        scheduleSidebarRedraw({ immediate: true });
      } else {
        state.pendingViewportState = preservedViewportState;
        openActiveImageLayer();
      }
    }
  } catch (error) {
    if (loadToken === state.slideLoadToken) {
      setTopbarStatus("error");
      setViewportEditorStatus(String(error));
    }
    throw error;
  } finally {
    if (loadToken === state.slideLoadToken) {
      populateSlideSelector();
      populateExperimentSelector(slideKey);
      diagnosisSelect.disabled = false;
      infiltrationSelect.disabled = false;
      experimentSelect.disabled = false;
      if (getFilteredSlides().length > 0) {
        slideSelect.value = slideKey;
      }
      syncSlideNavigationButtons();
    }
  }
}

function syncSlideQuery(slideKey) {
  syncUiStateQuery(slideKey);
}

function initializeViewer() {
  const initialTileSource = dziAssetUrl(IMAGE_LAYER_TILE_SOURCES[state.activeImageLayer]);
  const viewportMargins = getViewerViewportMargins();
  state.viewer = OpenSeadragon({
    id: "viewer",
    prefixUrl: window.SILICA_PORTAL.openSeadragonPrefix,
    tileSources: initialTileSource,
    showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT",
    toolbarPosition: "BOTTOM_LEFT",
    viewportMargins,
    visibilityRatio: 1,
    constrainDuringPan: true,
    animationTime: 0.7,
    blendTime: 0.1,
    minZoomLevel: 0.5,
    maxZoomPixelRatio: 2.5,
    zoomPerScroll: 1.25,
  });
  attachOverlayCanvasToViewer();

  state.viewer.addHandler("open", () => {
    state.pendingImageLayerFallback = null;
    setTopbarStatus("ready");
    restoreOrInitializeViewportState();
    applyPendingDotRadiusValue();
    syncUiStateQuery();
    scheduleRedraw();
    scheduleSidebarRedraw({ immediate: true });
  });
  state.viewer.addHandler("open-failed", () => {
    if (state.pendingImageLayerFallback !== null) {
      state.activeImageLayer = state.pendingImageLayerFallback;
      state.pendingImageLayerFallback = null;
      syncImageLayerControls();
      openActiveImageLayer();
      return;
    }
    setTopbarStatus("error");
  });
  state.viewer.addHandler("tile-load-failed", (event) => {
    const tileUrl = event?.tile?.url || event?.src || "unknown tile";
    setTopbarStatus("error");
    setViewportEditorStatus(`Failed to load DZI tile: ${tileUrl}`);
  });
  state.viewer.addHandler("animation", () => {
    scheduleRedraw();
    scheduleSidebarRedraw();
    scheduleViewportQuerySync();
  });
  state.viewer.addHandler("resize", () => {
    scheduleRedraw();
    scheduleSidebarRedraw();
    scheduleViewportQuerySync();
  });
  state.viewer.addHandler("pan", () => {
    scheduleRedraw();
    scheduleSidebarRedraw();
    scheduleViewportQuerySync();
  });
  state.viewer.addHandler("zoom", () => {
    scheduleRedraw();
    scheduleSidebarRedraw();
    scheduleViewportQuerySync();
  });
}

function attachOverlayCanvasToViewer() {
  const overlayCanvas = document.getElementById("overlayCanvas");
  const viewerContainer = document.querySelector("#viewer .openseadragon-container");
  if (!overlayCanvas || !viewerContainer) {
    throw new Error("Unable to attach overlay canvas to OpenSeadragon viewer");
  }
  viewerContainer.appendChild(overlayCanvas);
}

function getViewerViewportMargins() {
  return {
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  };
}

function updateViewerViewportMargins() {
  if (!state.viewer) {
    return;
  }
  state.viewer.viewportMargins = getViewerViewportMargins();
  state.viewer.viewport?.applyConstraints();
}

function setActiveImageLayer(imageLayer) {
  if (!Object.prototype.hasOwnProperty.call(IMAGE_LAYER_TILE_SOURCES, imageLayer)) {
    throw new Error(`Unknown image layer: ${imageLayer}`);
  }
  if (state.activeImageLayer === imageLayer) {
    return;
  }
  state.pendingImageLayerFallback = state.activeImageLayer;
  state.activeImageLayer = imageLayer;
  syncImageLayerControls();
  syncUiStateQuery();
  openActiveImageLayer({ preserveViewport: true });
}

function syncImageLayerControls() {
  for (const button of document.querySelectorAll(".image-layer-button")) {
    const isActive = button.dataset.imageLayer === state.activeImageLayer;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-checked", isActive ? "true" : "false");
  }
}

function openActiveImageLayer({ preserveViewport = false } = {}) {
  if (!state.viewer || !state.currentSlideKey) {
    return;
  }
  if (preserveViewport) {
    state.pendingViewportState = captureViewportState();
  }
  const tileSource = slideDziAssetUrl(
    state.currentSlideKey,
    IMAGE_LAYER_TILE_SOURCES[state.activeImageLayer],
  );
  setTopbarStatus("loading");
  state.viewer.open(tileSource);
}

function setTopbarStatus(status) {
  if (!["loading", "ready", "error"].includes(status)) {
    throw new Error(`Unknown topbar status: ${status}`);
  }
  const statusPill = document.querySelector(".topbar-filter-pill");
  if (!statusPill) {
    return;
  }
  if (state.topbarStatusHideTimer !== null) {
    window.clearTimeout(state.topbarStatusHideTimer);
    state.topbarStatusHideTimer = null;
  }
  state.topbarStatus = status;
  statusPill.classList.toggle("is-working", status === "loading");
  statusPill.classList.toggle("is-ready", status === "ready");
  statusPill.classList.toggle("is-error", status === "error");
  if (status === "ready") {
    state.topbarStatusHideTimer = window.setTimeout(() => {
      statusPill.classList.remove("is-working", "is-ready", "is-error");
      state.topbarStatus = "idle";
      state.topbarStatusHideTimer = null;
    }, 1000);
  }
}

function captureViewportState() {
  if (!state.viewer || !state.viewer.viewport || !state.manifest) {
    return null;
  }
  const visibleImageRect = getVisibleImageRect();
  const centerX = clampNumber(
    visibleImageRect.x + visibleImageRect.width / 2,
    0,
    state.manifest.image_width,
  );
  const centerY = clampNumber(
    visibleImageRect.y + visibleImageRect.height / 2,
    0,
    state.manifest.image_height,
  );
  return {
    center: {
      x: centerX / state.manifest.image_width,
      y: centerY / state.manifest.image_height,
    },
    zoom: clampNumber(
      visibleImageRect.height / state.manifest.image_height,
      1.0e-6,
      1,
    ),
  };
}

function restoreViewportState() {
  if (!state.pendingViewportState) {
    return;
  }
  if (
    !state.viewer?.viewport ||
    !state.manifest ||
    typeof state.viewer.viewport.imageToViewportRectangle !== "function" ||
    typeof state.viewer.viewport.fitBounds !== "function"
  ) {
    return;
  }
  const { center, zoom } = state.pendingViewportState;
  const containerSize = state.viewer.viewport.getContainerSize();
  const containerWidth = Math.max(1, containerSize.x);
  const containerHeight = Math.max(1, containerSize.y);
  const heightRatio = clampNumber(zoom, 1.0e-6, 1);
  const imageHeight = state.manifest.image_height * heightRatio;
  const imageWidth = imageHeight * (containerWidth / containerHeight);
  const centerImageX = clampNumber(
    center.x,
    0,
    1,
  ) * state.manifest.image_width;
  const centerImageY = clampNumber(
    center.y,
    0,
    1,
  ) * state.manifest.image_height;
  const viewportRect = state.viewer.viewport.imageToViewportRectangle(
    centerImageX - imageWidth / 2,
    centerImageY - imageHeight / 2,
    imageWidth,
    imageHeight,
  );
  state.viewer.viewport.fitBounds(viewportRect, true);
  state.viewer.viewport.applyConstraints();
  state.pendingViewportState = null;
}

function restoreOrInitializeViewportState() {
  if (state.pendingViewportState) {
    restoreViewportState();
    return;
  }
  state.pendingViewportState = {
    center: {
      x: DEFAULT_VIEWPORT_CENTER_X,
      y: DEFAULT_VIEWPORT_CENTER_Y,
    },
    zoom: DEFAULT_VIEWPORT_HEIGHT_RATIO,
  };
  restoreViewportState();
}

function scheduleRedraw() {
  if (state.redrawPending) {
    return;
  }
  state.redrawPending = true;
  window.requestAnimationFrame(() => {
    state.redrawPending = false;
    redrawScene();
  });
}

function scheduleSidebarRedraw(options = {}) {
  const { immediate = false } = options;
  if (state.sidebarRedrawTimer !== null) {
    window.clearTimeout(state.sidebarRedrawTimer);
    state.sidebarRedrawTimer = null;
  }
  if (immediate) {
    if (state.sidebarRedrawPending) {
      return;
    }
    state.sidebarRedrawPending = true;
    window.requestAnimationFrame(() => {
      state.sidebarRedrawPending = false;
      redrawSidebarPanels();
    });
    return;
  }

  state.sidebarRedrawTimer = window.setTimeout(() => {
    state.sidebarRedrawTimer = null;
    if (state.sidebarRedrawPending) {
      return;
    }
    state.sidebarRedrawPending = true;
    window.requestAnimationFrame(() => {
      state.sidebarRedrawPending = false;
      redrawSidebarPanels();
    });
  }, SIDEBAR_REDRAW_DELAY_MS);
}

function redrawScene() {
  if (!state.viewer || !state.viewer.world || state.viewer.world.getItemCount() === 0) {
    return;
  }

  const overlayCanvas = document.getElementById("overlayCanvas");
  const overlayContext = overlayCanvas.getContext("2d");
  const { width: overlayWidth, height: overlayHeight } = resizeCanvas(overlayCanvas);
  overlayContext.clearRect(0, 0, overlayWidth, overlayHeight);

  const visible = collectVisibleCells();
  drawCellOverlay(overlayContext, visible);
}

function redrawSidebarPanels() {
  if (!state.viewer || !state.viewer.world || state.viewer.world.getItemCount() === 0) {
    return;
  }

  const viewportCells = collectVisibleCells({ applyClusterFilter: false });
  const visible = collectVisibleCells();
  updateViewportSummary(visible, viewportCells);
  drawScoreHistogram(viewportCells.indices);
  drawClusterHistogram(viewportCells.indices);
  const viewportState = captureViewportState();
  if (viewportState) {
    updateViewportInputs(viewportState);
  }
}

function collectVisibleCells(options = {}) {
  const { applyClusterFilter = true } = options;
  const imageRect = getVisibleImageRect();
  const minX = imageRect.x;
  const maxX = imageRect.x + imageRect.width;
  const minY = imageRect.y;
  const maxY = imageRect.y + imageRect.height;
  const indices = [];

  for (let index = 0; index < state.cells.x.length; index += 1) {
    const x = state.cells.x[index];
    const y = state.cells.y[index];
    if (
      applyClusterFilter &&
      !state.activeClusterFilters.has(state.cells.dominant_cluster[index])
    ) {
      continue;
    }
    if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
      indices.push(index);
    }
  }

  return {
    indices,
    bounds: imageRect,
  };
}

function getVisibleImageRect() {
  const viewportBounds = state.viewer.viewport.getBounds(true);
  let imageRect = viewportBounds;
  if (typeof state.viewer.viewport.viewportToImageRectangle === "function") {
    imageRect = state.viewer.viewport.viewportToImageRectangle(viewportBounds);
  }

  const x = clampNumber(imageRect.x, 0, state.manifest.image_width);
  const y = clampNumber(imageRect.y, 0, state.manifest.image_height);
  const width = clampNumber(
    imageRect.width,
    0,
    state.manifest.image_width - x,
  );
  const height = clampNumber(
    imageRect.height,
    0,
    state.manifest.image_height - y,
  );
  return { x, y, width, height };
}

function getScreenPixelsPerImagePixel() {
  if (
    !state.viewer ||
    !state.viewer.viewport ||
    !state.manifest ||
    !state.viewer.world ||
    state.viewer.world.getItemCount() === 0
  ) {
    return null;
  }
  const visibleRect = getVisibleImageRect();
  const containerSize = state.viewer.viewport.getContainerSize();
  const containerHeight = Math.max(1, containerSize.y);
  const visibleHeight = Math.max(visibleRect.height, 1.0e-6);
  return containerHeight / visibleHeight;
}

function getRenderedDotRadius() {
  if (state.pendingDotRadiusValue !== null) {
    const screenPixelsPerImagePixel = getScreenPixelsPerImagePixel();
    if (Number.isFinite(screenPixelsPerImagePixel) && screenPixelsPerImagePixel > 0) {
      return clampNumber(
        (state.pendingDotRadiusValue * screenPixelsPerImagePixel) / 2,
        0.25,
        64,
      );
    }
    return clampNumber(state.pendingDotRadiusValue / 2, 0.25, 64);
  }
  const screenPixelsPerImagePixel = getScreenPixelsPerImagePixel();
  if (
    !Number.isFinite(screenPixelsPerImagePixel) ||
    screenPixelsPerImagePixel <= 0 ||
    !Number.isFinite(state.dotDiameterImagePx)
  ) {
    return clampNumber(DEFAULT_DOT_DIAMETER_IMAGE_PX / 2, 0.25, 64);
  }
  return clampNumber(
    (state.dotDiameterImagePx * screenPixelsPerImagePixel) / 2,
    0.25,
    64,
  );
}

function updateDotSizeControl(dotDiameterImagePx) {
  const slider = document.getElementById("dotSizeSlider");
  const valueLabel = document.getElementById("dotSizeValue");
  if (!slider || !valueLabel) {
    return;
  }
  slider.min = `${DOT_DIAMETER_IMAGE_MIN}`;
  slider.max = `${DOT_DIAMETER_IMAGE_MAX}`;
  const clampedDiameter = clampNumber(
    dotDiameterImagePx,
    DOT_DIAMETER_IMAGE_MIN,
    DOT_DIAMETER_IMAGE_MAX,
  );
  slider.value = clampedDiameter.toFixed(1);
  valueLabel.textContent = `${clampedDiameter.toFixed(1)}px`;
}

function updateBoxSizeControl(boxSizeImagePx) {
  const slider = document.getElementById("boxSizeSlider");
  const valueLabel = document.getElementById("boxSizeValue");
  if (!slider || !valueLabel) {
    return;
  }
  slider.min = `${BOX_SIZE_IMAGE_MIN}`;
  slider.max = `${BOX_SIZE_IMAGE_MAX}`;
  const clampedSize = clampNumber(
    boxSizeImagePx,
    BOX_SIZE_IMAGE_MIN,
    BOX_SIZE_IMAGE_MAX,
  );
  slider.value = clampedSize.toFixed(1);
  valueLabel.textContent = `${clampedSize.toFixed(1)}px`;
}

function updatePartitionAlphaControl(alpha) {
  const slider = document.getElementById("partitionAlphaSlider");
  const valueLabel = document.getElementById("partitionAlphaValue");
  if (!slider || !valueLabel) {
    return;
  }
  const clampedAlpha = clampNumber(alpha, 0, 1);
  slider.value = `${Math.round(clampedAlpha * 100)}`;
  valueLabel.textContent = `${Math.round(clampedAlpha * 100)}%`;
}

function updateBinaryThresholdControl(threshold) {
  const slider = document.getElementById("binaryThresholdSlider");
  const valueLabel = document.getElementById("binaryThresholdValue");
  const scaleBar = document.getElementById("scoreScaleBar");
  if (!slider || !valueLabel) {
    return;
  }
  const clampedThreshold = clampNumber(threshold, 0, 1);
  const thresholdPercent = Math.round(clampedThreshold * 100);
  slider.value = `${thresholdPercent}`;
  valueLabel.textContent = `${thresholdPercent}%`;
  valueLabel.style.setProperty("--binary-threshold-percent", `${thresholdPercent}%`);
  if (scaleBar) {
    scaleBar.style.setProperty(
      "--binary-threshold-percent",
      `${thresholdPercent}%`,
    );
  }
}

function applyPendingDotRadiusValue() {
  if (state.pendingDotRadiusValue === null) {
    return;
  }
  const clampedDiameter = clampNumber(
    state.pendingDotRadiusValue,
    DOT_DIAMETER_IMAGE_MIN,
    DOT_DIAMETER_IMAGE_MAX,
  );
  state.dotDiameterImagePx = clampedDiameter;
  updateDotSizeControl(clampedDiameter);
  state.pendingDotRadiusValue = null;
}

function applyPendingBoxSizeValue() {
  if (state.pendingBoxSizeValue === null) {
    return;
  }
  const clampedSize = clampNumber(
    state.pendingBoxSizeValue,
    BOX_SIZE_IMAGE_MIN,
    BOX_SIZE_IMAGE_MAX,
  );
  state.boxSizeImagePx = clampedSize;
  updateBoxSizeControl(clampedSize);
  state.pendingBoxSizeValue = null;
}

function applyPendingClusterFilters() {
  if (state.pendingClusterFilterIds === null) {
    return;
  }

  const allClusterIds = getAllClusterIds();
  if (state.pendingClusterFilterIds === "all") {
    state.activeClusterFilters = new Set(allClusterIds);
    state.pendingClusterFilterIds = null;
    return;
  }
  if (state.pendingClusterFilterIds === "none") {
    state.activeClusterFilters = new Set();
    state.pendingClusterFilterIds = null;
    return;
  }

  const validClusterIds = state.pendingClusterFilterIds.filter((clusterId) =>
    allClusterIds.includes(clusterId),
  );
  state.activeClusterFilters =
    validClusterIds.length > 0 ? new Set(validClusterIds) : new Set(allClusterIds);
  state.pendingClusterFilterIds = null;
}

function syncOverlayControlVisibility() {
  const dotSizeCard = document.getElementById("dotSizeCard");
  const boxSizeCard = document.getElementById("boxSizeCard");
  const partitionAlphaCard = document.getElementById("partitionAlphaCard");
  const binaryThresholdSlider = document.getElementById("binaryThresholdSlider");
  const binaryThresholdValue = document.getElementById("binaryThresholdValue");
  const scaleBar = document.getElementById("scoreScaleBar");
  const showDots = document.getElementById("toggleDots")?.checked ?? false;
  const showBoxes = document.getElementById("toggleBoxes")?.checked ?? false;
  const showPartitionFill =
    document.getElementById("togglePartitionFill")?.checked ?? false;
  const showBinaryDots =
    document.getElementById("toggleBinaryDots")?.checked ?? false;
  if (dotSizeCard) {
    dotSizeCard.hidden = !showDots;
  }
  if (boxSizeCard) {
    boxSizeCard.hidden = !showBoxes;
  }
  if (partitionAlphaCard) {
    partitionAlphaCard.hidden = !showPartitionFill;
  }
  if (binaryThresholdSlider) {
    binaryThresholdSlider.hidden = !showBinaryDots;
  }
  if (binaryThresholdValue) {
    binaryThresholdValue.hidden = !showBinaryDots;
  }
  if (scaleBar) {
    scaleBar.classList.toggle("is-binary", showBinaryDots);
  }
}

function setViewportEditorStatus(message) {
  const status = document.getElementById("viewportEditorStatus");
  if (!status) {
    return;
  }
  status.textContent = message || "";
}

function updateViewportInputs(bounds) {
  const inputVx = document.getElementById("viewportInputVx");
  const inputVy = document.getElementById("viewportInputVy");
  const inputVz = document.getElementById("viewportInputVz");
  if (!inputVx || !inputVy || !inputVz) {
    return;
  }
  inputVx.value = `${bounds.center.x.toFixed(2)}`;
  inputVy.value = `${bounds.center.y.toFixed(2)}`;
  inputVz.value = `${bounds.zoom.toFixed(2)}`;
  setViewportEditorStatus("");
}

function applyViewportBoundsFromInputs() {
  if (!state.viewer?.viewport || !state.manifest) {
    return;
  }

  const inputVx = document.getElementById("viewportInputVx");
  const inputVy = document.getElementById("viewportInputVy");
  const inputVz = document.getElementById("viewportInputVz");
  const rawVx = Number(inputVx?.value);
  const rawVy = Number(inputVy?.value);
  const rawVz = Number(inputVz?.value);
  if (
    !Number.isFinite(rawVx) ||
    !Number.isFinite(rawVy) ||
    !Number.isFinite(rawVz)
  ) {
    setViewportEditorStatus("Enter numeric Center X, Center Y, and View Height values.");
    return;
  }
  if (rawVz <= 0) {
    setViewportEditorStatus("View Height must be greater than 0.");
    return;
  }

  state.pendingViewportState = {
    center: {
      x: clampNumber(rawVx, 0, 1),
      y: clampNumber(rawVy, 0, 1),
    },
    zoom: clampNumber(rawVz, 1.0e-6, 1),
  };
  restoreViewportState();
  updateViewportInputs(captureViewportState() || state.pendingViewportState);
  setViewportEditorStatus("Viewport updated.");
  syncUiStateQuery();
  scheduleRedraw();
  scheduleSidebarRedraw({ immediate: true });
}

function scheduleViewportAutoApply(options = {}) {
  const { immediate = false } = options;
  if (state.viewportApplyTimer !== null) {
    window.clearTimeout(state.viewportApplyTimer);
    state.viewportApplyTimer = null;
  }
  if (immediate) {
    applyViewportBoundsFromInputs();
    return;
  }
  setViewportEditorStatus("Applying...");
  state.viewportApplyTimer = window.setTimeout(() => {
    state.viewportApplyTimer = null;
    applyViewportBoundsFromInputs();
  }, VIEWPORT_AUTO_APPLY_DELAY_MS);
}

function getOverlayToggleConfig() {
  return [
    { id: "toggleDots", key: "dots" },
    { id: "toggleBoxes", key: "boxes" },
    { id: "togglePartitionFill", key: "partition" },
    { id: "toggleScoreText", key: "score" },
    { id: "toggleClusterText", key: "cluster" },
    { id: "toggleContributionText", key: "contrib" },
  ];
}

function applyUiStateFromQuery() {
  const searchParams = new URLSearchParams(window.location.search);
  state.pendingDotRadiusValue = DEFAULT_DOT_DIAMETER_IMAGE_PX;
  state.pendingBoxSizeValue = DEFAULT_BOX_SIZE_IMAGE_PX;
  state.partitionFillAlpha = DEFAULT_PARTITION_FILL_ALPHA;
  state.binaryDotThreshold = DEFAULT_BINARY_DOT_SCORE_THRESHOLD;
  const requestedExperiment = (
    searchParams.get(EXPERIMENT_QUERY_PARAM) ??
    searchParams.get(LEGACY_EXPERIMENT_QUERY_PARAM) ??
    ""
  ).trim();
  if (requestedExperiment) {
    state.pendingExperiment = requestedExperiment;
  }
  let requestedOverlays = null;
  const overlaysRaw = searchParams.get(OVERLAY_QUERY_PARAM);
  if (overlaysRaw !== null) {
    requestedOverlays = new Set(
      overlaysRaw
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
    for (const overlay of getOverlayToggleConfig()) {
      const input = document.getElementById(overlay.id);
      if (input) {
        input.checked = requestedOverlays.has(overlay.key);
      }
    }
  }

  const viewRaw = (searchParams.get(VIEW_QUERY_PARAM) ?? "").trim();
  const binaryDotsInput = document.getElementById("toggleBinaryDots");
  if (binaryDotsInput) {
    if (viewRaw === VIEW_MODE_BINARY) {
      binaryDotsInput.checked = true;
    } else if (viewRaw === VIEW_MODE_CONTINUOUS) {
      binaryDotsInput.checked = false;
    } else if (requestedOverlays?.has("binary")) {
      binaryDotsInput.checked = true;
    }
  }

  const dotSizeRawValue = searchParams.get(DOT_SIZE_QUERY_PARAM);
  const dotSizeRaw =
    dotSizeRawValue === null ? Number.NaN : Number(dotSizeRawValue);
  if (Number.isFinite(dotSizeRaw)) {
    state.pendingDotRadiusValue = clampNumber(
      dotSizeRaw,
      DOT_DIAMETER_IMAGE_MIN,
      DOT_DIAMETER_IMAGE_MAX,
    );
  }

  const boxSizeRawValue = searchParams.get(BOX_SIZE_QUERY_PARAM);
  const boxSizeRaw =
    boxSizeRawValue === null ? Number.NaN : Number(boxSizeRawValue);
  if (Number.isFinite(boxSizeRaw)) {
    state.pendingBoxSizeValue = clampNumber(
      boxSizeRaw,
      BOX_SIZE_IMAGE_MIN,
      BOX_SIZE_IMAGE_MAX,
    );
  }

  const alphaRawValue = searchParams.get(PARTITION_ALPHA_QUERY_PARAM);
  const alphaRaw =
    alphaRawValue === null ? Number.NaN : Number(alphaRawValue);
  if (requestedOverlays?.has("partition") && Number.isFinite(alphaRaw)) {
    state.partitionFillAlpha = clampNumber(alphaRaw / 100, 0, 1);
  }

  const thresholdRawValue = searchParams.get(BINARY_THRESHOLD_QUERY_PARAM);
  const thresholdRaw =
    thresholdRawValue === null ? Number.NaN : Number(thresholdRawValue);
  if (Number.isFinite(thresholdRaw)) {
    state.binaryDotThreshold = clampNumber(thresholdRaw / 100, 0, 1);
  }

  const clustersRaw = searchParams.get(CLUSTER_QUERY_PARAM);
  if (clustersRaw === "all" || clustersRaw === "none") {
    state.pendingClusterFilterIds = clustersRaw;
  } else if (clustersRaw) {
    const clusterIds = clustersRaw
      .split(",")
      .map((value) => Number.parseInt(value, 10))
      .filter((value) => Number.isInteger(value));
    state.pendingClusterFilterIds = clusterIds;
  }

  const imageLayerRaw = searchParams.get(IMAGE_LAYER_QUERY_PARAM);
  if (Object.prototype.hasOwnProperty.call(IMAGE_LAYER_TILE_SOURCES, imageLayerRaw)) {
    state.activeImageLayer = imageLayerRaw;
  }

  const viewportXRaw = Number(searchParams.get(VIEWPORT_X_QUERY_PARAM));
  const viewportYRaw = Number(searchParams.get(VIEWPORT_Y_QUERY_PARAM));
  const viewportZoomRaw = Number(searchParams.get(VIEWPORT_ZOOM_QUERY_PARAM));
  if (
    Number.isFinite(viewportXRaw) &&
    Number.isFinite(viewportYRaw) &&
    Number.isFinite(viewportZoomRaw) &&
    viewportZoomRaw > 0
  ) {
    state.pendingViewportState = {
      center: { x: viewportXRaw, y: viewportYRaw },
      zoom: viewportZoomRaw,
    };
  }
}

function syncUiStateQuery(slideKey = state.currentSlideKey) {
  const url = new URL(window.location.href);
  const enabledOverlays = getOverlayToggleConfig()
    .filter((overlay) => document.getElementById(overlay.id)?.checked)
    .map((overlay) => overlay.key);
  const partitionFillEnabled = enabledOverlays.includes("partition");
  const binaryViewEnabled =
    document.getElementById("toggleBinaryDots")?.checked ?? false;
  const dotSizeValue = clampNumber(
    state.pendingDotRadiusValue ?? state.dotDiameterImagePx ?? DEFAULT_DOT_DIAMETER_IMAGE_PX,
    DOT_DIAMETER_IMAGE_MIN,
    DOT_DIAMETER_IMAGE_MAX,
  );
  const boxSizeValue = clampNumber(
    state.pendingBoxSizeValue ?? state.boxSizeImagePx ?? DEFAULT_BOX_SIZE_IMAGE_PX,
    BOX_SIZE_IMAGE_MIN,
    BOX_SIZE_IMAGE_MAX,
  );
  const partitionAlphaValue = Math.round(
    clampNumber(
      partitionFillEnabled ? state.partitionFillAlpha : DEFAULT_PARTITION_FILL_ALPHA,
      0,
      1,
    ) * 100,
  );
  const binaryThresholdValue = Math.round(
    clampNumber(state.binaryDotThreshold, 0, 1) * 100,
  );

  const allClusterIds = state.cells ? getAllClusterIds() : [];
  let clusterQueryValue = "all";
  if (state.cells) {
    if (state.activeClusterFilters.size === 0) {
      clusterQueryValue = "none";
    } else if (
      allClusterIds.length > 0 &&
      allClusterIds.every((clusterId) => state.activeClusterFilters.has(clusterId))
    ) {
      clusterQueryValue = "all";
    } else {
      clusterQueryValue = Array.from(state.activeClusterFilters)
        .sort((left, right) => left - right)
        .join(",");
    }
  } else if (Array.isArray(state.pendingClusterFilterIds)) {
    clusterQueryValue = [...state.pendingClusterFilterIds]
      .sort((left, right) => left - right)
      .join(",");
  } else if (state.pendingClusterFilterIds === "none") {
    clusterQueryValue = "none";
  }

  const viewportState = captureViewportState();
  const nextSearchParams = new URLSearchParams();
  if (slideKey) {
    nextSearchParams.set("slide", slideKey);
  }
  if (state.currentExperiment) {
    nextSearchParams.set(EXPERIMENT_QUERY_PARAM, state.currentExperiment);
  }
  nextSearchParams.set(IMAGE_LAYER_QUERY_PARAM, state.activeImageLayer);
  nextSearchParams.set(OVERLAY_QUERY_PARAM, enabledOverlays.join(","));
  nextSearchParams.set(DOT_SIZE_QUERY_PARAM, dotSizeValue.toFixed(1));
  nextSearchParams.set(BOX_SIZE_QUERY_PARAM, boxSizeValue.toFixed(1));
  nextSearchParams.set(PARTITION_ALPHA_QUERY_PARAM, `${partitionAlphaValue}`);
  if (viewportState) {
    nextSearchParams.set(VIEWPORT_X_QUERY_PARAM, `${viewportState.center.x.toFixed(5)}`);
    nextSearchParams.set(VIEWPORT_Y_QUERY_PARAM, `${viewportState.center.y.toFixed(5)}`);
    nextSearchParams.set(VIEWPORT_ZOOM_QUERY_PARAM, `${viewportState.zoom.toFixed(5)}`);
  }
  nextSearchParams.set(
    VIEW_QUERY_PARAM,
    binaryViewEnabled ? VIEW_MODE_BINARY : VIEW_MODE_CONTINUOUS,
  );
  nextSearchParams.set(BINARY_THRESHOLD_QUERY_PARAM, `${binaryThresholdValue}`);
  nextSearchParams.set(CLUSTER_QUERY_PARAM, clusterQueryValue);
  url.search = nextSearchParams.toString();

  window.history.replaceState({}, "", url);
}

function scheduleViewportQuerySync() {
  if (state.viewportQuerySyncTimer !== null) {
    window.clearTimeout(state.viewportQuerySyncTimer);
  }
  state.viewportQuerySyncTimer = window.setTimeout(() => {
    state.viewportQuerySyncTimer = null;
    syncUiStateQuery();
  }, 120);
}

function imageRectToScreenRect(centerX, centerY, width, height) {
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  const topLeft = imageToScreenPoint(centerX - halfWidth, centerY - halfHeight);
  const bottomRight = imageToScreenPoint(centerX + halfWidth, centerY + halfHeight);
  if (!topLeft || !bottomRight) {
    return null;
  }
  return {
    x: topLeft.x,
    y: topLeft.y,
    width: bottomRight.x - topLeft.x,
    height: bottomRight.y - topLeft.y,
  };
}

function normalizeScreenRect(rect) {
  const left = Math.min(rect.x, rect.x + rect.width);
  const right = Math.max(rect.x, rect.x + rect.width);
  const top = Math.min(rect.y, rect.y + rect.height);
  const bottom = Math.max(rect.y, rect.y + rect.height);
  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  };
}

function ensureMinimumScreenRectSize(rect, centerPoint, minSize) {
  const normalized = normalizeScreenRect(rect);
  const width = Math.max(normalized.width, minSize);
  const height = Math.max(normalized.height, minSize);
  return {
    x: centerPoint.x - width / 2,
    y: centerPoint.y - height / 2,
    width,
    height,
  };
}

function drawCellOverlay(context, visible) {
  const showDots = document.getElementById("toggleDots").checked;
  const showBinaryDots = document.getElementById("toggleBinaryDots").checked;
  const showBoxes = document.getElementById("toggleBoxes").checked;
  const showPartitionFill = document.getElementById("togglePartitionFill").checked;
  const showScoreText = document.getElementById("toggleScoreText").checked;
  const showClusterText = document.getElementById("toggleClusterText").checked;
  const showContributionText = document.getElementById(
    "toggleContributionText",
  ).checked;
  const viewWidthRatio = visible.bounds.width / state.manifest.image_width;

  const renderScoreText =
    showScoreText &&
    visible.indices.length <= TEXT_RENDER_LIMIT &&
    viewWidthRatio <= SCORE_LABEL_VIEW_WIDTH_RATIO;
  const renderClusterText =
    showClusterText &&
    visible.indices.length <= TEXT_RENDER_LIMIT &&
    viewWidthRatio <= CLUSTER_LABEL_VIEW_WIDTH_RATIO;
  const renderContributionText =
    showContributionText &&
    visible.indices.length <= TEXT_RENDER_LIMIT &&
    viewWidthRatio <= CONTRIBUTION_LABEL_VIEW_WIDTH_RATIO;
  applyPendingDotRadiusValue();
  applyPendingBoxSizeValue();
  const dotRadius = getRenderedDotRadius();
  const boxSizeImagePx =
    state.pendingBoxSizeValue ?? state.boxSizeImagePx ?? DEFAULT_BOX_SIZE_IMAGE_PX;
  const dotStrokeWidth = 1.6;
  updateDotSizeControl(
    state.pendingDotRadiusValue ?? state.dotDiameterImagePx ?? DEFAULT_DOT_DIAMETER_IMAGE_PX,
  );
  updateBoxSizeControl(boxSizeImagePx);

  if (showPartitionFill) {
    drawPartitionOverlay(context, viewWidthRatio, visible.bounds, {
      useBinaryColors: showBinaryDots,
    });
  }

  context.textAlign = "center";
  context.textBaseline = "middle";
  context.font = "11px 'Roboto Mono', monospace";

  for (const index of visible.indices) {
    const point = imageToScreenPoint(state.cells.x[index], state.cells.y[index]);
    if (!point) {
      continue;
    }

    if (showDots) {
      const dotColor = getDotOverlayColor(index, showBinaryDots);
      context.save();
      context.globalAlpha = showBinaryDots ? 0.74 : 0.5;
      context.beginPath();
      context.fillStyle = dotColor;
      context.arc(point.x, point.y, dotRadius, 0, Math.PI * 2);
      context.fill();
      context.restore();
      context.lineWidth = showBinaryDots ? 1.2 : dotStrokeWidth;
      context.strokeStyle = showBinaryDots ? BINARY_DOT_STROKE_STYLE : dotColor;
      context.stroke();
    }

    if (showBoxes) {
      const boxRect = imageRectToScreenRect(
        state.cells.x[index],
        state.cells.y[index],
        boxSizeImagePx,
        boxSizeImagePx,
      );
      if (boxRect) {
        const visibleBoxRect = ensureMinimumScreenRectSize(
          boxRect,
          point,
          MIN_SCREEN_BOX_SIZE_PX,
        );
        context.lineWidth = 2;
        context.strokeStyle = OVERLAY_BOX_STROKE_STYLE;
        context.strokeRect(
          visibleBoxRect.x,
          visibleBoxRect.y,
          visibleBoxRect.width,
          visibleBoxRect.height,
        );
      }
    }

    const textLines = [];
    if (renderScoreText) {
      textLines.push(`${state.cells.tumor_score_display[index]}`);
    }
    if (renderClusterText) {
      textLines.push(`C${state.cells.dominant_cluster[index]}`);
    }
    if (renderContributionText) {
      textLines.push(`${state.cells.dominant_cluster_display[index]}%`);
    }
    if (!textLines.length) {
      continue;
    }

    for (let lineIndex = 0; lineIndex < textLines.length; lineIndex += 1) {
      const textY = point.y - 12 - lineIndex * 12;
      context.strokeStyle = OVERLAY_TEXT_STROKE_STYLE;
      context.lineWidth = 3;
      context.strokeText(textLines[lineIndex], point.x, textY);
      context.fillStyle = OVERLAY_TEXT_FILL_STYLE;
      context.fillText(textLines[lineIndex], point.x, textY);
    }
  }
}

function getDotOverlayColor(index, useBinaryColors) {
  if (!useBinaryColors) {
    return state.cells.dot_color[index];
  }
  const tumorScore = Number(state.cells.tumor_score[index]);
  const binaryScore =
    Number.isFinite(tumorScore) && tumorScore >= state.binaryDotThreshold ? 1 : 0;
  return getTumorScoreColor(binaryScore);
}

function drawPartitionOverlay(context, viewWidthRatio, visibleImageBounds, options = {}) {
  const { useBinaryColors = false } = options;
  const overlayCanvas = document.getElementById("overlayCanvas");
  const overlayWidth = overlayCanvas?.getBoundingClientRect().width ?? 0;
  const overlayHeight = overlayCanvas?.getBoundingClientRect().height ?? 0;
  if (overlayWidth <= 0 || overlayHeight <= 0) {
    return;
  }

  const candidatePoints = getPartitionCandidateScreenPoints(
    visibleImageBounds,
    overlayWidth,
    overlayHeight,
    { useBinaryColors },
  );
  if (candidatePoints.length === 0) {
    return;
  }

  const sampleStep =
    viewWidthRatio >= PARTITION_HIGH_RES_VIEW_WIDTH_RATIO
      ? PARTITION_ZOOMED_OUT_SAMPLE_STEP_PX
      : PARTITION_SAMPLE_STEP_PX;
  const columns = Math.max(1, Math.ceil(overlayWidth / sampleStep));
  const rows = Math.max(1, Math.ceil(overlayHeight / sampleStep));
  const buckets = buildPartitionBuckets(candidatePoints, PARTITION_BUCKET_SIZE_PX);
  const partitionCanvas = document.createElement("canvas");
  partitionCanvas.width = columns;
  partitionCanvas.height = rows;
  const partitionContext = partitionCanvas.getContext("2d", { alpha: true });
  if (!partitionContext) {
    return;
  }

  context.save();
  context.globalAlpha = state.partitionFillAlpha;
  partitionContext.clearRect(0, 0, columns, rows);
  for (let row = 0; row < rows; row += 1) {
    const sampleY = Math.min(overlayHeight - 0.5, (row + 0.5) * sampleStep);
    for (let column = 0; column < columns; column += 1) {
      const sampleX = Math.min(overlayWidth - 0.5, (column + 0.5) * sampleStep);
      const sampleImagePoint = screenToImagePoint(sampleX, sampleY);
      if (
        !sampleImagePoint ||
        sampleImagePoint.x < 0 ||
        sampleImagePoint.x > state.manifest.image_width ||
        sampleImagePoint.y < 0 ||
        sampleImagePoint.y > state.manifest.image_height
      ) {
        continue;
      }
      const nearestPoint = findNearestPartitionPoint(
        sampleX,
        sampleY,
        candidatePoints,
        buckets,
        PARTITION_BUCKET_SIZE_PX,
      );
      if (!nearestPoint) {
        continue;
      }
      partitionContext.fillStyle = nearestPoint.color;
      partitionContext.fillRect(column, row, 1, 1);
    }
  }
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(partitionCanvas, 0, 0, columns, rows, 0, 0, overlayWidth, overlayHeight);
  context.restore();
}

function getPartitionCandidateScreenPoints(
  visibleImageBounds,
  overlayWidth,
  overlayHeight,
  options = {},
) {
  const { useBinaryColors = false } = options;
  const expandedImageBounds = expandImageRectForScreenMargin(
    visibleImageBounds,
    overlayWidth,
    overlayHeight,
    PARTITION_VIEW_MARGIN_PX,
  );
  const points = [];
  for (let index = 0; index < state.cells.x.length; index += 1) {
    if (!state.activeClusterFilters.has(state.cells.dominant_cluster[index])) {
      continue;
    }
    const imageX = state.cells.x[index];
    const imageY = state.cells.y[index];
    if (
      imageX < expandedImageBounds.x ||
      imageX > expandedImageBounds.x + expandedImageBounds.width ||
      imageY < expandedImageBounds.y ||
      imageY > expandedImageBounds.y + expandedImageBounds.height
    ) {
      continue;
    }
    const point = imageToScreenPoint(state.cells.x[index], state.cells.y[index]);
    if (!point) {
      continue;
    }
    if (
      point.x < -PARTITION_VIEW_MARGIN_PX ||
      point.x > overlayWidth + PARTITION_VIEW_MARGIN_PX ||
      point.y < -PARTITION_VIEW_MARGIN_PX ||
      point.y > overlayHeight + PARTITION_VIEW_MARGIN_PX
    ) {
      continue;
    }
    points.push({
      x: point.x,
      y: point.y,
      color: getDotOverlayColor(index, useBinaryColors),
    });
  }
  return points;
}

function expandImageRectForScreenMargin(imageRect, overlayWidth, overlayHeight, screenMarginPx) {
  const safeOverlayWidth = Math.max(1, overlayWidth);
  const safeOverlayHeight = Math.max(1, overlayHeight);
  const imageMarginX = (imageRect.width / safeOverlayWidth) * screenMarginPx;
  const imageMarginY = (imageRect.height / safeOverlayHeight) * screenMarginPx;
  const x = clampNumber(imageRect.x - imageMarginX, 0, state.manifest.image_width);
  const y = clampNumber(imageRect.y - imageMarginY, 0, state.manifest.image_height);
  const maxX = clampNumber(
    imageRect.x + imageRect.width + imageMarginX,
    0,
    state.manifest.image_width,
  );
  const maxY = clampNumber(
    imageRect.y + imageRect.height + imageMarginY,
    0,
    state.manifest.image_height,
  );
  return {
    x,
    y,
    width: Math.max(0, maxX - x),
    height: Math.max(0, maxY - y),
  };
}

function buildPartitionBuckets(points, bucketSize) {
  const buckets = new Map();
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const bucketX = Math.floor(point.x / bucketSize);
    const bucketY = Math.floor(point.y / bucketSize);
    const bucketKey = `${bucketX},${bucketY}`;
    const bucket = buckets.get(bucketKey);
    if (bucket) {
      bucket.push(index);
    } else {
      buckets.set(bucketKey, [index]);
    }
  }
  return buckets;
}

function findNearestPartitionPoint(sampleX, sampleY, points, buckets, bucketSize) {
  const baseBucketX = Math.floor(sampleX / bucketSize);
  const baseBucketY = Math.floor(sampleY / bucketSize);
  let bestPoint = null;
  let bestDistanceSquared = Number.POSITIVE_INFINITY;
  const maxRing = 64;

  for (let ring = 0; ring <= maxRing; ring += 1) {
    const minBucketX = baseBucketX - ring;
    const maxBucketX = baseBucketX + ring;
    const minBucketY = baseBucketY - ring;
    const maxBucketY = baseBucketY + ring;

    for (let bucketX = minBucketX; bucketX <= maxBucketX; bucketX += 1) {
      for (let bucketY = minBucketY; bucketY <= maxBucketY; bucketY += 1) {
        if (
          ring > 0 &&
          bucketX > minBucketX &&
          bucketX < maxBucketX &&
          bucketY > minBucketY &&
          bucketY < maxBucketY
        ) {
          continue;
        }
        const bucket = buckets.get(`${bucketX},${bucketY}`);
        if (!bucket) {
          continue;
        }
        for (const pointIndex of bucket) {
          const point = points[pointIndex];
          const dx = point.x - sampleX;
          const dy = point.y - sampleY;
          const distanceSquared = dx * dx + dy * dy;
          if (distanceSquared < bestDistanceSquared) {
            bestDistanceSquared = distanceSquared;
            bestPoint = point;
          }
        }
      }
    }

    if (bestPoint) {
      const maxGuaranteedDistance = Math.max(0, ring) * bucketSize;
      if (bestDistanceSquared <= maxGuaranteedDistance * maxGuaranteedDistance) {
        break;
      }
    }
  }

  return bestPoint;
}

function screenToImagePoint(x, y) {
  if (
    !state.viewer ||
    !state.viewer.viewport ||
    typeof state.viewer.viewport.viewerElementToImageCoordinates !== "function"
  ) {
    return null;
  }

  const imagePoint = state.viewer.viewport.viewerElementToImageCoordinates(
    new OpenSeadragon.Point(x, y),
  );
  return { x: imagePoint.x, y: imagePoint.y };
}

function imageToScreenPoint(x, y) {
  if (
    !state.viewer ||
    !state.viewer.viewport ||
    typeof state.viewer.viewport.imageToViewerElementCoordinates !== "function"
  ) {
    return null;
  }

  const osdPoint = state.viewer.viewport.imageToViewerElementCoordinates(
    new OpenSeadragon.Point(x, y),
  );
  return { x: osdPoint.x, y: osdPoint.y };
}

function updateViewportSummary(visible, viewportCells = visible) {
  const visibleCount = viewportCells.indices.length;
  document.getElementById("visibleCellCount").textContent =
    formatInteger(visibleCount);

  if (!visibleCount) {
    const meanTumorScoreElement = document.getElementById("meanTumorScore");
    meanTumorScoreElement.textContent = "-";
    meanTumorScoreElement.style.color = "";
  } else {
    const meanTumorScore = meanTumorScoreForIndices(viewportCells.indices);
    setPercentMetric("meanTumorScore", meanTumorScore);
  }

  const clusterCounts = new Map();
  for (const index of viewportCells.indices) {
    const cluster = state.cells.dominant_cluster[index];
    clusterCounts.set(cluster, (clusterCounts.get(cluster) || 0) + 1);
  }

  if (!clusterCounts.size) {
    document.getElementById("topCluster").textContent = "-";
    return;
  }

  let topCluster = null;
  let topClusterCount = -1;
  for (const [cluster, count] of clusterCounts.entries()) {
    if (count > topClusterCount) {
      topCluster = cluster;
      topClusterCount = count;
    }
  }

  document.getElementById("topCluster").textContent = `C${topCluster} (${topClusterCount})`;
}

function getTumorScoreColor(tumorScore) {
  const clampedScore = clampNumber(tumorScore, 0, 1);

  let leftStop = TUMOR_SCORE_COLOR_STOPS[0];
  let rightStop = TUMOR_SCORE_COLOR_STOPS[TUMOR_SCORE_COLOR_STOPS.length - 1];
  for (let index = 1; index < TUMOR_SCORE_COLOR_STOPS.length; index += 1) {
    if (clampedScore <= TUMOR_SCORE_COLOR_STOPS[index].position) {
      leftStop = TUMOR_SCORE_COLOR_STOPS[index - 1];
      rightStop = TUMOR_SCORE_COLOR_STOPS[index];
      break;
    }
  }

  const segmentWidth = Math.max(1.0e-6, rightStop.position - leftStop.position);
  const t = (clampedScore - leftStop.position) / segmentWidth;
  const rgb = leftStop.color.map((channel, index) =>
    Math.round(channel + (rightStop.color[index] - channel) * t),
  );
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function drawScoreHistogram(indices) {
  const canvas = document.getElementById("scoreHistogram");
  const { width, height } = resizeCanvas(canvas);
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  const histogram = buildHistogram(
    indices.map((index) => state.cells.tumor_score[index]),
    18,
    0,
    1,
  );
  drawBarChart({
    context,
    width,
    height,
    xTickLabels: histogram.map((bin) => `${Math.round(bin.start * 100)}`),
    values: histogram.map((bin) => bin.count),
    fillStyle: SCORE_HISTOGRAM_FILL_STYLE,
    emptyLabel: "No visible cells",
  });
}

function drawClusterHistogram(indices) {
  const canvas = document.getElementById("clusterHistogram");
  const { width, height } = resizeCanvas(canvas);
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  const clusterCounts = Array.from(
    { length: state.manifest.num_clusters },
    () => 0,
  );
  for (const index of indices) {
    clusterCounts[state.cells.dominant_cluster[index]] += 1;
  }
  drawBarChart({
    context,
    width,
    height,
    xTickLabels: clusterCounts.map((_, index) => `${index}`),
    values: clusterCounts,
    fillStyle: CLUSTER_HISTOGRAM_FILL_STYLE,
    emptyLabel: "No visible cells",
  });
}

function drawBarChart({
  context,
  width,
  height,
  xTickLabels,
  values,
  fillStyle,
  emptyLabel,
}) {
  context.save();
  context.fillStyle = CHART_BACKGROUND_STYLE;
  context.fillRect(0, 0, width, height);

  context.font = "11px 'Roboto Mono', monospace";
  const maxValue = Math.max(...values, 0);
  const yAxisLabelWidth = Math.ceil(
    context.measureText(formatChartTick(maxValue)).width,
  );
  const padding = {
    top: 10,
    right: 8,
    bottom: 34,
    left: Math.max(28, yAxisLabelWidth + 12),
  };
  const chartWidth = Math.max(0, width - padding.left - padding.right);
  const chartHeight = Math.max(0, height - padding.top - padding.bottom);
  if (chartWidth === 0 || chartHeight === 0) {
    context.restore();
    return;
  }

  const yTickCount = maxValue <= 4 ? Math.max(2, maxValue) : 4;

  context.strokeStyle = CHART_GRID_STROKE_STYLE;
  context.lineWidth = 1;
  for (let tickIndex = 0; tickIndex <= yTickCount; tickIndex += 1) {
    const tickValue = (maxValue * tickIndex) / yTickCount;
    const y = padding.top + chartHeight - (chartHeight * tickIndex) / yTickCount;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + chartWidth, y);
    context.stroke();

    context.fillStyle = CHART_TEXT_FILL_STYLE;
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(formatChartTick(tickValue), padding.left - 6, y);
  }

  context.strokeStyle = CHART_AXIS_STROKE_STYLE;
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + chartHeight);
  context.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  context.stroke();

  if (maxValue === 0) {
    context.fillStyle = CHART_TEXT_FILL_STYLE;
    context.font = "500 13px Roboto, sans-serif";
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(emptyLabel, padding.left + 8, padding.top + chartHeight / 2);
    context.restore();
    return;
  }

  const barCount = values.length;
  const sideInset = barCount > 24 ? 1 : 3;
  const gap = barCount > 30 ? 1 : barCount > 14 ? 2 : 3;
  const plotWidth = Math.max(0, chartWidth - sideInset * 2);
  const barWidth = Math.max(
    1,
    (plotWidth - gap * Math.max(0, barCount - 1)) / barCount,
  );

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const barHeight = (value / maxValue) * chartHeight;
    const x = padding.left + sideInset + index * (barWidth + gap);
    const y = padding.top + chartHeight - barHeight;
    context.fillStyle = fillStyle;
    context.fillRect(x, y, barWidth, barHeight);
  }

  context.fillStyle = CHART_TEXT_FILL_STYLE;
  context.font = "11px 'Roboto Mono', monospace";
  context.textAlign = "center";
  context.textBaseline = "top";
  const tickStep = barCount > 18 ? Math.ceil(barCount / 10) : 1;
  for (let index = 0; index < xTickLabels.length; index += tickStep) {
    const label = xTickLabels[index];
    const x =
      padding.left + sideInset + index * (barWidth + gap) + barWidth / 2;
    const axisY = padding.top + chartHeight;
    context.fillText(label, x, axisY + 8);
  }
  context.restore();
}

function formatChartTick(value) {
  if (value >= 100) {
    return `${Math.round(value)}`;
  }
  if (value >= 10) {
    return `${Math.round(value)}`;
  }
  if (value === 0) {
    return "0";
  }
  return `${Math.round(value * 10) / 10}`;
}

function buildHistogram(values, binCount, minValue, maxValue) {
  const bins = Array.from({ length: binCount }, (_, index) => ({
    start: minValue + ((maxValue - minValue) * index) / binCount,
    end: minValue + ((maxValue - minValue) * (index + 1)) / binCount,
    count: 0,
  }));
  if (!values.length) {
    return bins;
  }

  const span = maxValue - minValue;
  for (const rawValue of values) {
    const value = clampNumber(rawValue, minValue, maxValue - Number.EPSILON);
    const binIndex = Math.min(
      binCount - 1,
      Math.floor(((value - minValue) / span) * binCount),
    );
    bins[binIndex].count += 1;
  }
  return bins;
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const devicePixelRatio = window.devicePixelRatio || 1;
  const nextWidth = Math.max(1, Math.round(rect.width * devicePixelRatio));
  const nextHeight = Math.max(1, Math.round(rect.height * devicePixelRatio));
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  return { width: rect.width, height: rect.height };
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

function dziAssetUrl(relativePath) {
  return slideDziAssetUrl(state.currentSlideKey, relativePath);
}

function slidePortalAssetUrl(slideKey, experimentName, relativePath) {
  return new URL(
    relativePath,
    `${window.location.origin}${window.SILICA_PORTAL.portalAssetBaseUrl}/${encodeURIComponent(
      experimentName,
    )}/${encodeURIComponent(slideKey)}/`,
  ).toString();
}

function slideDziAssetUrl(slideKey, relativePath) {
  return new URL(
    relativePath,
    `${window.location.origin}${window.SILICA_PORTAL.dziAssetBaseUrl}/${encodeURIComponent(slideKey)}/`,
  ).toString();
}

function clampNumber(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function formatInteger(value) {
  return new Intl.NumberFormat("en-US").format(value);
}
