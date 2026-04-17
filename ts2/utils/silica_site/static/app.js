const state = {
  slides: [],
  filterOptions: {
    diagnosis: [],
    infiltration: [],
  },
  displayFilters: {
    diagnosis: "",
    infiltration: "",
  },
  activeFilters: {
    diagnosis: "",
    infiltration: "",
  },
  manifest: null,
  cells: null,
  viewer: null,
  currentSlideKey: null,
  navigatorVisible: true,
  navigatorHideTimer: null,
  topbarStatusHideTimer: null,
  pendingViewportState: null,
  redrawPending: false,
  slideLoadToken: 0,
  dotRadiusOffset: 0,
  activeClusterFilters: new Set(),
  topbarSelectionRevealed: false,
};

const TEXT_RENDER_LIMIT = 2200;
const SCORE_LABEL_VIEW_WIDTH_RATIO = 0.34;
const CLUSTER_LABEL_VIEW_WIDTH_RATIO = 0.28;
const CONTRIBUTION_LABEL_VIEW_WIDTH_RATIO = 0.24;
const FILTER_ALL_VALUE = "__all__";
const DOT_RADIUS_MIN = 1;
const DOT_RADIUS_MAX = 8;
const CELL_BOX_SIZE_PX = 48;
const MIN_SCREEN_BOX_SIZE_PX = 8;
const CLUSTER_FILTER_MIN_COUNT = 20;
const INITIAL_VIEW_ZOOM_MULTIPLIER = 1.18;
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
    setTopbarStatus("loading");
    document.getElementById("viewportBounds").textContent = String(error);
  });
});

async function bootstrapPortal() {
  setTopbarStatus("loading");
  const slidesPayload = await fetchJson(window.SILICA_PORTAL.slidesUrl);
  state.slides = normalizeSlidesPayload(slidesPayload.slides);
  state.filterOptions = resolveFilterOptions(slidesPayload.filters, state.slides);
  if (state.slides.length === 0) {
    throw new Error("No slides were returned by /api/slides");
  }
  populateFilterSelectors();
  populateSlideSelector();
  bindControls();
  await changeSlide(resolveInitialSlideKey(slidesPayload.default_slide_key), {
    syncEmptyFilters: true,
  });
  initializeViewer();
}

function revealTopbarSelectionLabels() {
  if (state.topbarSelectionRevealed) {
    return;
  }
  if (
    !state.activeFilters.diagnosis &&
    !state.activeFilters.infiltration &&
    state.displayFilters.diagnosis &&
    state.displayFilters.infiltration
  ) {
    state.activeFilters = { ...state.displayFilters };
    populateFilterSelectors();
    populateSlideSelector();
    if (state.currentSlideKey && getFilteredSlides().some((slide) => slide.key === state.currentSlideKey)) {
      document.getElementById("slideSelect").value = state.currentSlideKey;
    }
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
    diagnosis: normalizeFilterValue(slide.diagnosis),
    infiltration: normalizeFilterValue(slide.infiltration),
  }));
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
  document.getElementById("topbarClusterLabel").textContent = `GMM K ${formatInteger(
    state.cells.num_clusters,
  )}`;
  const slideMeanTumorScore =
    state.cells.tumor_score.reduce((sum, value) => sum + value, 0) /
    state.cells.tumor_score.length;
  const slideMeanTumorScoreElement = document.getElementById("slideMeanTumorScore");
  slideMeanTumorScoreElement.textContent = `${(
    slideMeanTumorScore * 100
  ).toFixed(1)}%`;
  slideMeanTumorScoreElement.style.color = getTumorScoreColor(slideMeanTumorScore);
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

function getClusterFilterGroups() {
  const sortedClusters = getSortedClustersByCount();
  const groups = [];
  const otherClusterIds = [];
  let otherCount = 0;

  for (const [cluster, count] of sortedClusters) {
    if (count < CLUSTER_FILTER_MIN_COUNT) {
      otherClusterIds.push(cluster);
      otherCount += count;
      continue;
    }
    groups.push({
      key: `cluster-${cluster}`,
      label: `C${cluster} (${formatInteger(count)})`,
      clusterIds: [cluster],
    });
  }

  if (otherClusterIds.length > 0) {
    groups.push({
      key: "cluster-others",
      label: `Others (${formatInteger(otherCount)})`,
      clusterIds: otherClusterIds,
    });
  }

  return groups;
}

function getAllClusterIds() {
  return getSortedClustersByCount().map(([cluster]) => cluster);
}

function areAllClustersSelected() {
  const allClusterIds = getAllClusterIds();
  return (
    allClusterIds.length > 0 &&
    allClusterIds.every((cluster) => state.activeClusterFilters.has(cluster))
  );
}

function populateClusterFilterControls() {
  const container = document.getElementById("clusterFilterControls");
  if (!container || !state.cells) {
    return;
  }

  const clusterGroups = getClusterFilterGroups();

  container.replaceChildren();

  const allLabel = document.createElement("label");
  allLabel.className = "toggle-chip cluster-filter-chip cluster-filter-chip-all";
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.dataset.cluster = "all";
  allInput.checked = areAllClustersSelected();
  const allText = document.createElement("span");
  allText.className = "toggle-label";
  allText.textContent = "All";
  allLabel.append(allInput, allText);
  container.appendChild(allLabel);

  for (const group of clusterGroups) {
    const label = document.createElement("label");
    label.className = "toggle-chip cluster-filter-chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.clusterGroup = group.key;
    input.checked = group.clusterIds.every((cluster) =>
      state.activeClusterFilters.has(cluster),
    );
    const text = document.createElement("span");
    text.className = "toggle-label";
    text.textContent = group.label;
    label.append(input, text);
    container.appendChild(label);
  }
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
    return;
  }
  slideSelect.value = matchingSlides[0].key;
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

  select.value = state.displayFilters[filterKey] || "";
}

function willFilterOptionChangeContext(filterKey, value) {
  const oppositeFilterKey =
    filterKey === "diagnosis" ? "infiltration" : "diagnosis";
  const oppositeFilterValue = state.displayFilters[oppositeFilterKey];
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
      void changeSlide(event.target.value, { syncEmptyFilters: true });
    }
  });
  document
    .getElementById("clusterFilterControls")
    .addEventListener("change", (event) => {
      const input = event.target.closest("input[type='checkbox']");
      if (!input) {
        return;
      }
      const rawClusterGroup = input.dataset.clusterGroup ?? input.dataset.cluster;
      if (rawClusterGroup === "all") {
        if (input.checked) {
          state.activeClusterFilters = new Set(getAllClusterIds());
        } else {
          state.activeClusterFilters = new Set();
        }
      } else {
        const group = getClusterFilterGroups().find(
          (candidate) => candidate.key === rawClusterGroup,
        );
        if (!group) {
          return;
        }
        const nextSelected = new Set(state.activeClusterFilters);
        if (input.checked) {
          for (const cluster of group.clusterIds) {
            nextSelected.add(cluster);
          }
        } else {
          for (const cluster of group.clusterIds) {
            nextSelected.delete(cluster);
          }
        }
        state.activeClusterFilters = nextSelected;
      }
      populateClusterFilterControls();
      scheduleRedraw();
    });

  for (const input of document.querySelectorAll("input[type='checkbox']")) {
    input.addEventListener("change", scheduleRedraw);
  }
  document.getElementById("dotSizeSlider").addEventListener("input", (event) => {
    const sliderValue = Number(event.target.value);
    if (!Number.isFinite(sliderValue)) {
      return;
    }
    const autoRadius = getAutoDotRadius(getCurrentViewWidthRatio());
    state.dotRadiusOffset = sliderValue - autoRadius;
    updateDotSizeControl(sliderValue);
    scheduleRedraw();
  });

  window.addEventListener("resize", scheduleRedraw);
}

function applyFilters(nextFilters) {
  const { filters, matchingSlides } = resolveFilterTransition(nextFilters);
  state.activeFilters = filters;
  state.displayFilters = { ...filters };
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
    void changeSlide(preferredSlideKey);
  }
}

async function changeSlide(slideKey, options = {}) {
  const { syncEmptyFilters = false } = options;
  if (!slideKey) {
    throw new Error("No slide is available for the current filter selection");
  }
  const slideEntry = getSlideEntry(slideKey);
  if (!slideEntry) {
    throw new Error(`Unknown slide key: ${slideKey}`);
  }

  if (
    syncEmptyFilters &&
    !state.activeFilters.diagnosis &&
    !state.activeFilters.infiltration
  ) {
    state.displayFilters = {
      diagnosis: slideEntry.diagnosis,
      infiltration: slideEntry.infiltration,
    };
    populateFilterSelectors();
  }
  state.activeClusterFilters = new Set();

  const slideSelect = document.getElementById("slideSelect");
  const diagnosisSelect = document.getElementById("diagnosisSelect");
  const infiltrationSelect = document.getElementById("infiltrationSelect");
  const loadToken = state.slideLoadToken + 1;
  state.slideLoadToken = loadToken;
  setTopbarStatus("loading");
  slideSelect.disabled = true;
  diagnosisSelect.disabled = true;
  infiltrationSelect.disabled = true;
  slideSelect.value = slideKey;

  try {
    const manifest = await fetchJson(slidePortalAssetUrl(slideKey, "slide_manifest.json"));
    const cells = await fetchJson(slidePortalAssetUrl(slideKey, manifest.cells.path));
    if (loadToken !== state.slideLoadToken) {
      return;
    }

    state.currentSlideKey = slideKey;
    state.manifest = manifest;
    state.cells = cells;
    state.activeClusterFilters = new Set(getAllClusterIds());
    populateSlideHeader();
    populateClusterFilterControls();
    syncSlideQuery(slideKey);

    if (state.viewer) {
      state.pendingViewportState = null;
      state.viewer.open(
        slideDziAssetUrl(slideKey, state.manifest.base_layers.color.dzi),
      );
    }
  } finally {
    if (loadToken === state.slideLoadToken) {
      populateSlideSelector();
      diagnosisSelect.disabled = false;
      infiltrationSelect.disabled = false;
      if (getFilteredSlides().length > 0) {
        slideSelect.value = slideKey;
      }
    }
  }
}

function syncSlideQuery(slideKey) {
  const url = new URL(window.location.href);
  url.searchParams.set("slide", slideKey);
  window.history.replaceState({}, "", url);
}

function initializeViewer() {
  const initialTileSource = dziAssetUrl(state.manifest.base_layers.color.dzi);
  state.viewer = OpenSeadragon({
    id: "viewer",
    prefixUrl: window.SILICA_PORTAL.openSeadragonPrefix,
    tileSources: initialTileSource,
    showNavigator: true,
    visibilityRatio: 1,
    constrainDuringPan: true,
    animationTime: 0.7,
    blendTime: 0.1,
    minZoomLevel: 0.5,
    maxZoomPixelRatio: 2.5,
    zoomPerScroll: 1.25,
  });

  state.viewer.addHandler("open", () => {
    setTopbarStatus("ready");
    markNavigatorVisible();
    restoreOrInitializeViewportState();
    scheduleRedraw();
  });
  state.viewer.addHandler("open-failed", () => {
    setTopbarStatus("loading");
  });
  state.viewer.addHandler("animation", () => {
    markNavigatorVisible();
    scheduleRedraw();
  });
  state.viewer.addHandler("resize", () => {
    markNavigatorVisible();
    scheduleRedraw();
  });
  state.viewer.addHandler("pan", () => {
    markNavigatorVisible();
    scheduleRedraw();
  });
  state.viewer.addHandler("zoom", () => {
    markNavigatorVisible();
    scheduleRedraw();
  });
}

function setTopbarStatus(status) {
  const statusPill = document.getElementById("topbarStatus");
  if (!statusPill) {
    return;
  }
  if (state.topbarStatusHideTimer !== null) {
    window.clearTimeout(state.topbarStatusHideTimer);
    state.topbarStatusHideTimer = null;
  }
  statusPill.classList.remove("is-hidden");
  statusPill.classList.toggle("is-working", status === "loading");
  statusPill.classList.toggle("is-ready", status === "ready");
  if (status === "ready") {
    state.topbarStatusHideTimer = window.setTimeout(() => {
      statusPill.classList.add("is-hidden");
      state.topbarStatusHideTimer = null;
    }, 5000);
  }
}

function captureViewportState() {
  if (!state.viewer || !state.viewer.viewport) {
    return null;
  }
  return {
    center: state.viewer.viewport.getCenter(true),
    zoom: state.viewer.viewport.getZoom(true),
  };
}

function restoreViewportState() {
  if (!state.pendingViewportState) {
    return;
  }
  const { center, zoom } = state.pendingViewportState;
  state.viewer.viewport.zoomTo(zoom, null, true);
  state.viewer.viewport.panTo(center, true);
  state.viewer.viewport.applyConstraints();
  state.pendingViewportState = null;
}

function applyInitialViewportZoom() {
  if (!state.viewer?.viewport || typeof state.viewer.viewport.getHomeZoom !== "function") {
    return;
  }
  const homeZoom = state.viewer.viewport.getHomeZoom();
  state.viewer.viewport.zoomTo(
    homeZoom * INITIAL_VIEW_ZOOM_MULTIPLIER,
    null,
    true,
  );
  state.viewer.viewport.applyConstraints();
}

function restoreOrInitializeViewportState() {
  if (state.pendingViewportState) {
    restoreViewportState();
    return;
  }
  applyInitialViewportZoom();
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

function redrawScene() {
  if (!state.viewer || !state.viewer.world || state.viewer.world.getItemCount() === 0) {
    return;
  }

  const overlayCanvas = document.getElementById("overlayCanvas");
  const overlayContext = overlayCanvas.getContext("2d");
  const { width: overlayWidth, height: overlayHeight } = resizeCanvas(overlayCanvas);
  overlayContext.clearRect(0, 0, overlayWidth, overlayHeight);

  const visible = collectVisibleCells();
  const viewportCells = collectVisibleCells({ applyClusterFilter: false });
  drawCellOverlay(overlayContext, visible);
  updateViewportSummary(visible, viewportCells);
  drawScoreHistogram(viewportCells.indices);
  drawClusterHistogram(viewportCells.indices);
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

function getCurrentViewWidthRatio() {
  if (
    !state.viewer ||
    !state.manifest ||
    !state.viewer.world ||
    state.viewer.world.getItemCount() === 0
  ) {
    return 1;
  }
  const visibleRect = getVisibleImageRect();
  return visibleRect.width / state.manifest.image_width;
}

function getAutoDotRadius(viewWidthRatio) {
  const safeViewWidthRatio = Math.max(viewWidthRatio, 1e-4);
  const zoomBoost = Math.log10(1 / safeViewWidthRatio);
  return clampNumber(4 + zoomBoost * 2.8, DOT_RADIUS_MIN, DOT_RADIUS_MAX);
}

function getDotRadius(viewWidthRatio) {
  return clampNumber(
    getAutoDotRadius(viewWidthRatio) + state.dotRadiusOffset,
    DOT_RADIUS_MIN,
    DOT_RADIUS_MAX,
  );
}

function updateDotSizeControl(dotRadius) {
  const slider = document.getElementById("dotSizeSlider");
  const valueLabel = document.getElementById("dotSizeValue");
  if (!slider || !valueLabel) {
    return;
  }
  const clampedRadius = clampNumber(dotRadius, DOT_RADIUS_MIN, DOT_RADIUS_MAX);
  slider.value = clampedRadius.toFixed(1);
  valueLabel.textContent = `${clampedRadius.toFixed(1)}px`;
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
  const showBoxes = document.getElementById("toggleBoxes").checked;
  const showScoreText = document.getElementById("toggleScoreText").checked;
  const showClusterText = document.getElementById("toggleClusterText").checked;
  const showContributionText = document.getElementById(
    "toggleContributionText",
  ).checked;
  const navigatorBounds = getNavigatorOverlayBounds();
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
  const dotRadius = getDotRadius(viewWidthRatio);
  const dotStrokeWidth = clampNumber(dotRadius * 0.26, 1.1, 1.8);
  updateDotSizeControl(dotRadius);

  context.textAlign = "center";
  context.textBaseline = "middle";
  context.font = "11px 'Roboto Mono', monospace";

  for (const index of visible.indices) {
    const point = imageToScreenPoint(state.cells.x[index], state.cells.y[index]);
    if (!point || isPointInsideRect(point, navigatorBounds)) {
      continue;
    }

    if (showDots) {
      context.beginPath();
      context.fillStyle = state.cells.dot_color[index];
      context.arc(point.x, point.y, dotRadius, 0, Math.PI * 2);
      context.fill();
      context.lineWidth = dotStrokeWidth;
      context.strokeStyle = "rgba(255, 255, 255, 0.72)";
      context.stroke();
    }

    if (showBoxes) {
      const boxRect = imageRectToScreenRect(
        state.cells.x[index],
        state.cells.y[index],
        CELL_BOX_SIZE_PX,
        CELL_BOX_SIZE_PX,
      );
      if (boxRect) {
        const visibleBoxRect = ensureMinimumScreenRectSize(
          boxRect,
          point,
          MIN_SCREEN_BOX_SIZE_PX,
        );
        context.lineWidth = 1.75;
        context.strokeStyle = state.cells.dot_color[index];
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
      context.strokeStyle = "rgba(255, 255, 255, 0.92)";
      context.lineWidth = 3;
      context.strokeText(textLines[lineIndex], point.x, textY);
      context.fillStyle = "#292524";
      context.fillText(textLines[lineIndex], point.x, textY);
    }
  }
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

function getNavigatorOverlayBounds() {
  if (!state.navigatorVisible) {
    return null;
  }

  const navigatorWrapper = getNavigatorControlWrapper();
  if (!navigatorWrapper) {
    return null;
  }

  const wrapperStyle = window.getComputedStyle(navigatorWrapper);
  const wrapperOpacity = Number.parseFloat(wrapperStyle.opacity || "1");
  if (
    wrapperStyle.display === "none" ||
    wrapperStyle.visibility === "hidden" ||
    wrapperOpacity < 0.05
  ) {
    return null;
  }

  const navigatorRect = navigatorWrapper.getBoundingClientRect();
  const overlayRect = document
    .getElementById("overlayCanvas")
    ?.getBoundingClientRect();
  if (!overlayRect || navigatorRect.width <= 0 || navigatorRect.height <= 0) {
    return null;
  }

  const padding = 10;
  return {
    left: navigatorRect.left - overlayRect.left - padding,
    top: navigatorRect.top - overlayRect.top - padding,
    right: navigatorRect.right - overlayRect.left + padding,
    bottom: navigatorRect.bottom - overlayRect.top + padding,
  };
}

function getNavigatorControlWrapper() {
  if (!state.viewer?.navigator?.element || !Array.isArray(state.viewer.controls)) {
    return null;
  }

  const navigatorElement = state.viewer.navigator.element;
  const navigatorControl = state.viewer.controls.find(
    (control) => control?.element === navigatorElement,
  );
  return navigatorControl?.wrapper || navigatorElement.parentElement || null;
}

function markNavigatorVisible() {
  state.navigatorVisible = true;
  if (state.navigatorHideTimer !== null) {
    window.clearTimeout(state.navigatorHideTimer);
  }

  const fadeDelay = Number(state.viewer?.controlsFadeDelay ?? 2000);
  const fadeLength = Number(state.viewer?.controlsFadeLength ?? 1500);
  const hideDelayMs = Math.max(0, fadeDelay) + Math.max(0, fadeLength);

  state.navigatorHideTimer = window.setTimeout(() => {
    state.navigatorVisible = false;
    state.navigatorHideTimer = null;
    scheduleRedraw();
  }, hideDelayMs);
}

function isPointInsideRect(point, rect) {
  if (!rect) {
    return false;
  }
  return (
    point.x >= rect.left &&
    point.x <= rect.right &&
    point.y >= rect.top &&
    point.y <= rect.bottom
  );
}

function updateViewportSummary(visible, viewportCells = visible) {
  const visibleCount = visible.indices.length;
  document.getElementById("visibleCellCount").textContent =
    formatInteger(visibleCount);
  document.getElementById("viewportBounds").textContent =
    `x=${Math.round(visible.bounds.x)} y=${Math.round(visible.bounds.y)} ` +
    `w=${Math.round(visible.bounds.width)} h=${Math.round(visible.bounds.height)}`;

  if (!visibleCount) {
    const meanTumorScoreElement = document.getElementById("meanTumorScore");
    meanTumorScoreElement.textContent = "-";
    meanTumorScoreElement.style.color = "";
  } else {
    let tumorScoreSum = 0;
    for (const index of visible.indices) {
      tumorScoreSum += state.cells.tumor_score[index];
    }

    const meanTumorScore = tumorScoreSum / visibleCount;
    const meanTumorScoreElement = document.getElementById("meanTumorScore");
    meanTumorScoreElement.textContent = `${(meanTumorScore * 100).toFixed(1)}%`;
    meanTumorScoreElement.style.color = getTumorScoreColor(meanTumorScore);
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
  const colorStops = [
    { position: 0, color: [26, 152, 80] },
    { position: 0.32, color: [217, 239, 139] },
    { position: 0.58, color: [253, 219, 199] },
    { position: 0.78, color: [239, 138, 98] },
    { position: 1, color: [178, 24, 43] },
  ];

  let leftStop = colorStops[0];
  let rightStop = colorStops[colorStops.length - 1];
  for (let index = 1; index < colorStops.length; index += 1) {
    if (clampedScore <= colorStops[index].position) {
      leftStop = colorStops[index - 1];
      rightStop = colorStops[index];
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
    fillStyle: "#2563eb",
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
    fillStyle: "#57534e",
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
  context.fillStyle = "#fafaf9";
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

  context.strokeStyle = "rgba(68, 64, 60, 0.12)";
  context.lineWidth = 1;
  for (let tickIndex = 0; tickIndex <= yTickCount; tickIndex += 1) {
    const tickValue = (maxValue * tickIndex) / yTickCount;
    const y = padding.top + chartHeight - (chartHeight * tickIndex) / yTickCount;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + chartWidth, y);
    context.stroke();

    context.fillStyle = "#78716c";
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(formatChartTick(tickValue), padding.left - 6, y);
  }

  context.strokeStyle = "rgba(68, 64, 60, 0.28)";
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + chartHeight);
  context.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  context.stroke();

  if (maxValue === 0) {
    context.fillStyle = "#78716c";
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

  context.fillStyle = "#78716c";
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

function slidePortalAssetUrl(slideKey, relativePath) {
  return new URL(
    relativePath,
    `${window.location.origin}${window.SILICA_PORTAL.portalAssetBaseUrl}/${encodeURIComponent(slideKey)}/`,
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
