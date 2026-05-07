const {
  ATTN_SCALE_MAX,
  ATTN_SCALE_MIN,
  BOX_SIZE_IMAGE_MAX,
  BOX_SIZE_IMAGE_MIN,
  DEFAULT_BOX_SIZE_IMAGE_PX,
  DEFAULT_DOT_DIAMETER_IMAGE_PX,
  DOT_DIAMETER_IMAGE_MAX,
  DOT_DIAMETER_IMAGE_MIN,
  IMAGE_LAYER_TILE_SOURCES,
  SIDEBAR_REDRAW_DELAY_MS,
  VIEW_MODE_ATTN,
  VIEW_MODE_BINARY,
  VIEW_MODE_CONTINUOUS,
} = SilicaSiteConfig;

const {
  clampNumber,
  formatInteger,
  resizeCanvas,
} = SilicaSiteUtils;

const {
  normalizeExperimentNames,
  normalizeSlidesPayload,
  resolveFilterOptions,
} = SilicaSlideModel;

const {
  getAllClusterIds: getAllClusterIdsForCells,
  getTopCluster: getTopClusterForCells,
  meanTumorScoreForIndices: meanTumorScoreForCellIndices,
} = SilicaSlideStats;

const {
  resolveClusterFilterInputValue,
  resolvePendingClusterFilterPatch,
  setClusterFilterStatus: setClusterFilterStatusElement,
  snapshotClusterFilterState: snapshotClusterFilterStateForCells,
  syncClusterFilterEditor: syncClusterFilterEditorForState,
} = SilicaSlideClusterFilter;

const {
  getTumorScoreColor,
} = SilicaSlideColors;

const {
  drawClusterHistogram: drawClusterHistogramChart,
  drawScoreHistogram: drawScoreHistogramChart,
} = SilicaSlideCharts;

const {
  drawCellOverlay: drawCellOverlayFromState,
} = SilicaSlideOverlayRenderer;

const {
  applyFilters: applyFiltersForNavigation,
  ensureValidCurrentExperiment: ensureValidCurrentExperimentForNavigation,
  getFilteredSlides: getFilteredSlidesFromNavigation,
  getSlideAvailableExperiments: getSlideAvailableExperimentsForNavigation,
  getSlideEntry: getSlideEntryForNavigation,
  navigateFilteredSlides: navigateFilteredSlidesForNavigation,
  populateExperimentSelector: populateExperimentSelectorForNavigation,
  populateFilterSelectors: populateFilterSelectorsForNavigation,
  populateSlideSelector: populateSlideSelectorForNavigation,
  resolveInitialExperiment: resolveInitialExperimentForNavigation,
  resolveInitialSlideKey: resolveInitialSlideKeyForNavigation,
  syncSlideNavigationButtons: syncSlideNavigationButtonsForNavigation,
} = SilicaSlideNavigation;

const {
  initializeViewer: initializeViewerFromState,
  openActiveImageLayer: openActiveImageLayerFromState,
  refreshViewerAfterLayoutResize: refreshViewerAfterLayoutResizeFromState,
  setActiveImageLayer: setActiveImageLayerFromState,
  syncImageLayerControls: syncImageLayerControlsForState,
} = SilicaSlideImageLayer;

const {
  loadSlideIndex,
  loadSlidePayloads,
} = SilicaSlideLoader;

const state = SilicaSlideState.createInitialState();

const els = {};

function bindElements() {
  Object.assign(els, SilicaSlideViewerElements.bind());
}

function commit(patch, effects = {}) {
  Object.assign(state, patch);

  if (effects.clusterEditor) {
    syncClusterFilterEditor();
  }
  if (effects.overlayControls) {
    syncOverlayControlVisibility();
  }
  if (effects.query) {
    syncUiStateQuery();
  }
  if (effects.redraw) {
    scheduleRedraw();
  }
  if (effects.sidebar) {
    scheduleSidebarRedraw({ immediate: effects.sidebar === "immediate" });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  void bootstrapPortal().catch((error) => {
    console.error(error);
    setTopbarStatus("error");
    setViewportEditorStatus(String(error));
  });
});

async function bootstrapPortal() {
  bindElements();
  setTopbarStatus("loading");
  applyUiStateFromQuery();
  const slidesPayload = await loadSlideIndex();
  const slides = normalizeSlidesPayload(slidesPayload.slides);
  commit({
    slides,
    availableExperiments: normalizeExperimentNames(slidesPayload.experiments),
    filterOptions: resolveFilterOptions(slidesPayload.filters, slides),
  });
  if (state.slides.length === 0) {
    throw new Error("No slides were returned by /api/slides");
  }
  const initialSlideKey = resolveInitialSlideKey(slidesPayload.default_slide_key);
  if (!getSlideEntry(initialSlideKey)) {
    commit({ pendingSlideKey: initialSlideKey });
  }
  commit({
    currentExperiment: resolveInitialExperiment(
      slidesPayload.default_experiment,
      initialSlideKey,
    ),
    pendingExperiment: null,
  });
  populateFilterSelectors();
  populateSlideSelector();
  populateExperimentSelector(initialSlideKey);
  bindControls();
  await changeSlide(initialSlideKey, {
    syncEmptyFilters: true,
  });
}

function revealTopbarSelectionLabels() {
  SilicaUI.revealTopbarSelectionLabels(state);
}

function populateSlideHeader() {
  els.totalCellCount.textContent = formatInteger(state.cells.cell_count);
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
  const element = els[elementId];
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
  const element = els[elementId];
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    element.textContent = "-";
    element.style.color = "";
    return;
  }

  element.textContent = `${(numericValue * 100).toFixed(1)}%`;
  element.style.color = getTumorScoreColor(numericValue);
}

function getAllClusterIds() {
  return getAllClusterIdsForCells(state.cells);
}

function setClusterFilterStatus(message) {
  setClusterFilterStatusElement(els.clusterFilterStatus, message);
}

function syncClusterFilterEditor() {
  syncClusterFilterEditorForState({
    input: els.clusterFilterInput,
    statusElement: els.clusterFilterStatus,
    cells: state.cells,
    pendingClusterFilterIds: state.pendingClusterFilterIds,
    activeClusterFilters: state.activeClusterFilters,
  });
}

function applyClusterFilterInputValue() {
  if (!state.cells) {
    return;
  }
  const input = els.clusterFilterInput;

  const allClusterIds = getAllClusterIds();
  const nextFilterState = resolveClusterFilterInputValue({
    rawValue: input.value,
    allClusterIds,
  });
  commit(
    { activeClusterFilters: nextFilterState.activeClusterFilters },
    { clusterEditor: true, query: true, redraw: true, sidebar: "immediate" },
  );
  if (nextFilterState.status) {
    setClusterFilterStatus(nextFilterState.status);
  }
}

function populateSlideSelector() {
  populateSlideSelectorForNavigation({ state, els });
}

function syncSlideNavigationButtons() {
  syncSlideNavigationButtonsForNavigation({ state, els });
}

function populateFilterSelectors() {
  populateFilterSelectorsForNavigation({ state, els });
}

function resolveInitialSlideKey(defaultSlideKey) {
  return resolveInitialSlideKeyForNavigation({
    state,
    defaultSlideKey,
    search: window.location.search,
  });
}

function getSlideEntry(slideKey) {
  return getSlideEntryForNavigation(state, slideKey);
}

function getSlideAvailableExperiments(slideKey) {
  return getSlideAvailableExperimentsForNavigation(state, slideKey);
}

function resolveInitialExperiment(defaultExperiment, slideKey) {
  return resolveInitialExperimentForNavigation({
    state,
    defaultExperiment,
    slideKey,
  });
}

function ensureValidCurrentExperiment(slideKey = state.currentSlideKey) {
  return ensureValidCurrentExperimentForNavigation({ state, commit, slideKey });
}

function populateExperimentSelector(slideKey = state.currentSlideKey) {
  populateExperimentSelectorForNavigation({ state, els, commit, slideKey });
}

function getFilteredSlides() {
  return getFilteredSlidesFromNavigation(state);
}

function bindControls() {
  SilicaSlideControls.bindControls({
    state,
    els,
    handlers: {
      applyClusterFilterInputValue,
      applyFilters,
      commit,
      getAttnScaleMaxFromSliderValue,
      getSlideAvailableExperiments,
      loadSlideFromUi,
      navigateFilteredSlides,
      populateExperimentSelector,
      refreshViewerAfterLayoutResize,
      revealTopbarSelectionLabels,
      scheduleRedraw,
      scheduleSidebarRedraw,
      scheduleViewportAutoApply,
      setActiveImageLayer,
      syncClusterFilterEditor,
      syncOverlayControlVisibility,
      syncUiStateQuery,
      updateAttnScaleControl,
      updateBinaryThresholdControl,
      updateBoxSizeControl,
      updateDotSizeControl,
      updatePartitionAlphaControl,
      updateViewerViewportMargins,
    },
  });
}

function normalizeOverlayCheckboxes(changedInput = null) {
  SilicaSlideControls.normalizeOverlayCheckboxes(els, changedInput);
}

function navigateFilteredSlides(direction) {
  navigateFilteredSlidesForNavigation({
    state,
    els,
    direction,
    syncUiStateQuery,
    loadSlideFromUi,
  });
}

function loadSlideFromUi(slideKey, options = {}) {
  void changeSlide(slideKey, options).catch((error) => {
    console.error(error);
    setTopbarStatus("error");
    setViewportEditorStatus(String(error));
  });
}

function refreshViewerAfterLayoutResize() {
  refreshViewerAfterLayoutResizeFromState({
    state,
    scheduleRedraw,
    scheduleSidebarRedraw,
    scheduleViewportQuerySync,
  });
}

function applyFilters(nextFilters) {
  applyFiltersForNavigation({
    state,
    els,
    commit,
    nextFilters,
    loadSlideFromUi,
  });
}

function snapshotClusterFilterState() {
  return snapshotClusterFilterStateForCells({
    cells: state.cells,
    activeClusterFilters: state.activeClusterFilters,
    pendingClusterFilterIds: state.pendingClusterFilterIds,
  });
}

function restorePendingClusterFilterState(clusterFilterState) {
  if (
    clusterFilterState === "all" ||
    clusterFilterState === "none" ||
    Array.isArray(clusterFilterState)
  ) {
    commit({ pendingClusterFilterIds: clusterFilterState });
    return;
  }
  commit({ pendingClusterFilterIds: null });
}

async function changeSlide(slideKey, options = {}) {
  const {
    syncEmptyFilters = false,
    preserveViewport = false,
    useAvailableExperiment = false,
  } = options;
  if (!slideKey) {
    throw new Error("No slide is available for the current filter selection");
  }
  setTopbarStatus("loading");
  commit({ pendingSlideKey: slideKey });

  const loadToken = state.slideLoadToken + 1;
  commit({ slideLoadToken: loadToken });
  els.slideSelect.disabled = true;
  els.diagnosisSelect.disabled = true;
  els.infiltrationSelect.disabled = true;
  els.experimentSelect.disabled = true;
  els.previousSlideButton.disabled = true;
  els.nextSlideButton.disabled = true;
  els.slideSelect.value = slideKey;
  populateExperimentSelector(slideKey);
  els.experimentSelect.disabled = true;

  try {
    const slideEntry = getSlideEntry(slideKey);
    if (!slideEntry) {
      throw new Error(`Unknown slide key: ${slideKey}`);
    }

    if (
      syncEmptyFilters &&
      (!state.activeFilters.diagnosis || !state.activeFilters.infiltration)
    ) {
      commit({
        activeFilters: {
          ...state.activeFilters,
          diagnosis: state.activeFilters.diagnosis
            ? state.activeFilters.diagnosis
            : slideEntry.diagnosis,
          infiltration: state.activeFilters.infiltration
            ? state.activeFilters.infiltration
            : slideEntry.infiltration,
        },
      });
      populateFilterSelectors();
      populateSlideSelector();
    }
    if (
      useAvailableExperiment &&
      state.currentExperiment &&
      !getSlideAvailableExperiments(slideKey).includes(state.currentExperiment)
    ) {
      commit({ currentExperiment: null });
    }
    const resolvedExperiment = ensureValidCurrentExperiment(slideKey);
    if (resolvedExperiment === null) {
      throw new Error(`No experiments are available for slide ${slideKey}`);
    }
    const preserveCurrentViewerImage =
      Boolean(state.viewer) && state.currentSlideKey === slideKey;
    const preservedViewportState =
      preserveViewport && !preserveCurrentViewerImage ? captureViewportState() : null;
    const preservedClusterFilterState = snapshotClusterFilterState();

    const { cells, manifest } = await loadSlidePayloads({
      experimentName: resolvedExperiment,
      slideKey,
    });
    if (loadToken !== state.slideLoadToken) {
      return;
    }

    commit({
      currentSlideKey: slideKey,
      currentExperiment: resolvedExperiment,
      pendingSlideKey: null,
      pendingExperiment: null,
      manifest,
      cells,
      ...getAttnScoreRangePatch(cells),
    });
    syncImageLayerControls();
    commit({ activeClusterFilters: new Set(getAllClusterIds()) });
    restorePendingClusterFilterState(preservedClusterFilterState);
    applyPendingAttnScaleMaxValue();
    applyPendingClusterFilters();
    populateSlideHeader();
    populateExperimentSelector(slideKey);
    syncClusterFilterEditor();
    syncOverlayControlVisibility();
    syncSlideQuery(slideKey);

    if (!state.viewer) {
      initializeViewer();
    } else if (preserveCurrentViewerImage) {
      setTopbarStatus("ready");
      commit({}, { redraw: true, sidebar: "immediate" });
    } else {
      commit({ pendingViewportState: preservedViewportState });
      openActiveImageLayer();
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
      els.diagnosisSelect.disabled = false;
      els.infiltrationSelect.disabled = false;
      if (getFilteredSlides().length > 0) {
        els.slideSelect.value = slideKey;
      }
      syncSlideNavigationButtons();
    }
  }
}

function syncSlideQuery(slideKey) {
  syncUiStateQuery(slideKey);
}

function initializeViewer() {
  initializeViewerFromState({
    state,
    commit,
    els,
    dziAssetUrl,
    handlers: {
      applyPendingDotRadiusValue,
      onViewportChanged() {
        scheduleRedraw();
        scheduleSidebarRedraw();
        scheduleViewportQuerySync();
      },
      openActiveImageLayer,
      restoreOrInitializeViewportState,
      scheduleRedraw,
      scheduleSidebarRedraw,
      setTopbarStatus,
      setViewportEditorStatus,
      syncImageLayerControls,
      syncUiStateQuery,
    },
  });
}

function updateViewerViewportMargins() {
  SilicaSlideViewport.updateViewerViewportMargins(state);
}

function setActiveImageLayer(imageLayer) {
  setActiveImageLayerFromState({
    state,
    commit,
    imageLayer,
    openActiveImageLayer,
    syncImageLayerControls,
  });
}

function syncImageLayerControls() {
  syncImageLayerControlsForState(state);
}

function openActiveImageLayer({ preserveViewport = false } = {}) {
  openActiveImageLayerFromState({
    state,
    commit,
    slideDziAssetUrl,
    setTopbarStatus,
    captureViewportState,
    preserveViewport,
  });
}

function setTopbarStatus(status) {
  SilicaUI.setStatusPill({
    state,
    status,
    pillSelector: ".topbar-filter-pill",
    statusName: "topbar",
  });
  if (status === "error" && !els.diagnosticsStatus.textContent) {
    setViewportEditorStatus("The slide viewer encountered an error.");
  }
}

function captureViewportState() {
  return SilicaSlideViewport.captureViewportState(state);
}

function restoreViewportState() {
  SilicaSlideViewport.restoreViewportState({ state, commit });
}

function restoreOrInitializeViewportState() {
  SilicaSlideViewport.restoreOrInitializeViewportState({ state, commit });
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

  const overlayContext = els.overlayCanvas.getContext("2d");
  const { width: overlayWidth, height: overlayHeight } = resizeCanvas(els.overlayCanvas);
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
  return SilicaSlideViewport.collectVisibleCells({
    state,
    applyClusterFilter: options.applyClusterFilter ?? true,
  });
}

function getScreenPixelsPerImagePixel() {
  return SilicaSlideViewport.getScreenPixelsPerImagePixel(state);
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
  els.dotSizeSlider.min = `${DOT_DIAMETER_IMAGE_MIN}`;
  els.dotSizeSlider.max = `${DOT_DIAMETER_IMAGE_MAX}`;
  const clampedDiameter = clampNumber(
    dotDiameterImagePx,
    DOT_DIAMETER_IMAGE_MIN,
    DOT_DIAMETER_IMAGE_MAX,
  );
  els.dotSizeSlider.value = clampedDiameter.toFixed(1);
  els.dotSizeValue.textContent = `${clampedDiameter.toFixed(1)}px`;
}

function updateBoxSizeControl(boxSizeImagePx) {
  els.boxSizeSlider.min = `${BOX_SIZE_IMAGE_MIN}`;
  els.boxSizeSlider.max = `${BOX_SIZE_IMAGE_MAX}`;
  const clampedSize = clampNumber(
    boxSizeImagePx,
    BOX_SIZE_IMAGE_MIN,
    BOX_SIZE_IMAGE_MAX,
  );
  els.boxSizeSlider.value = clampedSize.toFixed(1);
  els.boxSizeValue.textContent = `${clampedSize.toFixed(1)}px`;
}

function updatePartitionAlphaControl(alpha) {
  const clampedAlpha = clampNumber(alpha, 0, 1);
  els.partitionAlphaSlider.value = `${Math.round(clampedAlpha * 100)}`;
  els.partitionAlphaValue.textContent = `${Math.round(clampedAlpha * 100)}%`;
}

function updateBinaryThresholdControl(threshold) {
  const clampedThreshold = clampNumber(threshold, 0, 1);
  const thresholdPercent = Math.round(clampedThreshold * 100);
  els.binaryThresholdSlider.value = `${thresholdPercent}`;
  els.binaryThresholdValue.textContent = `${thresholdPercent}%`;
  els.binaryThresholdValue.style.setProperty(
    "--binary-threshold-percent",
    `${thresholdPercent}%`,
  );
  els.scoreScaleBar.style.setProperty(
    "--binary-threshold-percent",
    `${thresholdPercent}%`,
  );
}

function getAttnScaleMaxFromSliderValue(sliderValue) {
  if (state.attnScoreMin === null || state.attnScoreDataMax === null) {
    throw new Error("Attention score data range is unavailable");
  }
  const sliderPercent = clampNumber(sliderValue, 0, 100) / 100;
  return (
    ATTN_SCALE_MIN +
    (ATTN_SCALE_MAX - ATTN_SCALE_MIN) * sliderPercent
  );
}

function getAttnScaleSliderValue(attnScoreMax) {
  if (state.attnScoreMin === null || state.attnScoreDataMax === null) {
    throw new Error("Attention score data range is unavailable");
  }
  const range = ATTN_SCALE_MAX - ATTN_SCALE_MIN;
  if (range === 0) {
    return 0;
  }
  return clampNumber(((attnScoreMax - ATTN_SCALE_MIN) / range) * 100, 0, 100);
}

function updateAttnScaleControl() {
  if (
    state.attnScoreMin === null ||
    state.attnScoreMax === null ||
    state.attnScoreDataMax === null
  ) {
    els.attnScaleMaxSlider.disabled = true;
    els.attnScaleMinLabel.textContent = "-";
    els.attnScaleMaxValue.textContent = "-";
    els.attnScaleBar.style.setProperty("--attn-scale-max-percent", "90%");
    els.attnScaleMaxValue.style.setProperty("--attn-scale-max-percent", "90%");
    return;
  }
  const clampedMax = clampNumber(
    state.attnScoreMax,
    ATTN_SCALE_MIN,
    ATTN_SCALE_MAX,
  );
  const maxPercent = `${Math.round(getAttnScaleSliderValue(clampedMax))}%`;
  els.attnScaleMaxSlider.disabled = false;
  els.attnScaleMaxSlider.value = `${Math.round(getAttnScaleSliderValue(clampedMax))}`;
  els.attnScaleMinLabel.textContent = `${ATTN_SCALE_MIN}`;
  els.attnScaleMaxValue.textContent = clampedMax.toPrecision(3);
  els.attnScaleBar.style.setProperty("--attn-scale-max-percent", maxPercent);
  els.attnScaleMaxValue.style.setProperty("--attn-scale-max-percent", maxPercent);
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
  commit({
    dotDiameterImagePx: clampedDiameter,
    pendingDotRadiusValue: null,
  });
  updateDotSizeControl(clampedDiameter);
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
  commit({
    boxSizeImagePx: clampedSize,
    pendingBoxSizeValue: null,
  });
  updateBoxSizeControl(clampedSize);
}

function applyPendingAttnScaleMaxValue() {
  if (state.pendingAttnScaleMaxValue === null) {
    return;
  }
  if (state.attnScoreMin === null || state.attnScoreDataMax === null) {
    throw new Error("Attention score data range is unavailable");
  }
  commit({
    attnScoreMax: clampNumber(
      state.pendingAttnScaleMaxValue,
      ATTN_SCALE_MIN,
      ATTN_SCALE_MAX,
    ),
    pendingAttnScaleMaxValue: null,
  });
}

function applyPendingClusterFilters() {
  const pendingFilterPatch = resolvePendingClusterFilterPatch({
    cells: state.cells,
    pendingClusterFilterIds: state.pendingClusterFilterIds,
  });
  if (pendingFilterPatch) {
    commit(pendingFilterPatch);
  }
}

function getAttnScoreRangePatch(cells) {
  if (!Array.isArray(cells?.attn_score) || cells.attn_score.length === 0) {
    return { attnScoreMin: null, attnScoreMax: null, attnScoreDataMax: null };
  }

  const attnScores = [];
  let attnScoreMin = Number.POSITIVE_INFINITY;
  let attnScoreDataMax = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < cells.attn_score.length; index += 1) {
    const value = Number(cells.attn_score[index]);
    if (!Number.isFinite(value)) {
      throw new Error(`Invalid attention score at index ${index}: ${cells.attn_score[index]}`);
    }
    attnScores.push(value);
    attnScoreMin = Math.min(attnScoreMin, value);
    attnScoreDataMax = Math.max(attnScoreDataMax, value);
  }
  attnScores.sort((left, right) => left - right);
  const attnScoreMax = attnScores[Math.floor((attnScores.length - 1) * 0.9)];

  return { attnScoreMin, attnScoreMax, attnScoreDataMax };
}

function syncOverlayControlVisibility() {
  const hasCells = Boolean(state.cells);
  const hasAttnData = Array.isArray(state.cells?.attn_score) && state.cells.attn_score.length > 0;
  els.toggleAttnDots.disabled = !hasAttnData;
  els.dotModeAttnChip.classList.toggle("is-disabled", !hasAttnData);
  els.dotModeAttnChip.setAttribute("aria-disabled", String(!hasAttnData));
  if (hasCells && !hasAttnData && els.toggleAttnDots.checked) {
    els.toggleAttnDots.checked = false;
    els.dotModeContinuous.checked = true;
  }
  normalizeOverlayCheckboxes();
  const showDots = els.toggleDots.checked;
  const showAttnDots = els.toggleAttnDots.checked && hasAttnData;
  const showBoxes = els.toggleBoxes.checked;
  const showPartitionFill = els.togglePartitionFill.checked;
  const showBinaryDots = els.toggleBinaryDots.checked && showDots;
  els.dotSizeCard.hidden = !showDots && !showAttnDots;
  els.dotModeCard.hidden = !showDots;
  els.attributionModeCard.hidden = !els.toggleAttributionText.checked;
  els.attributionHint.hidden = !els.toggleAttributionText.checked;
  els.boxSizeCard.hidden = !showBoxes;
  els.partitionAlphaCard.hidden = !showPartitionFill;
  els.overlayScalePanel.hidden = false;
  els.scoreScaleStrip.hidden = !showDots || showAttnDots;
  els.scoreScaleBar.classList.toggle("is-binary", showBinaryDots);
  els.attnScaleStrip.hidden = !showDots || !showAttnDots;
  els.binaryThresholdSlider.hidden = !showBinaryDots;
  els.binaryThresholdValue.hidden = !showBinaryDots;
  syncOverlaySettingDividers();
  updateAttnScaleControl();
}

function syncOverlaySettingDividers() {
  let hasVisibleGroup = false;
  const settingsBox = document.querySelector(".overlay-settings-box");
  for (const group of document.querySelectorAll(".overlay-setting-group")) {
    const isVisible = Boolean(
      group.querySelector(".slider-card:not([hidden]), .hint-box:not([hidden])"),
    );
    group.classList.toggle("is-separated", isVisible && hasVisibleGroup);
    if (isVisible) {
      hasVisibleGroup = true;
    }
  }
  settingsBox.hidden = !hasVisibleGroup;
}

function setViewportEditorStatus(message) {
  SilicaSlideViewport.setViewportEditorStatus(els, message);
}

function updateViewportInputs(bounds) {
  SilicaSlideViewport.updateViewportInputs(els, bounds);
}

function applyViewportBoundsFromInputs() {
  SilicaSlideViewport.applyViewportBoundsFromInputs({
    state,
    els,
    commit,
    onApplied() {
      commit({}, { query: true, redraw: true, sidebar: "immediate" });
    },
  });
}

function scheduleViewportAutoApply(options = {}) {
  SilicaSlideViewport.scheduleViewportAutoApply({
    state,
    els,
    immediate: options.immediate ?? false,
    applyViewportBoundsFromInputs,
  });
}

function getOverlayToggleConfig() {
  return [
    { element: els.toggleDots, key: "dots" },
    { element: els.toggleBoxes, key: "boxes" },
    { element: els.togglePartitionFill, key: "partition" },
    { element: els.toggleAttributionText, key: "attribution" },
  ];
}

function getAttributionModes() {
  return [
    { element: els.attributionModeScore, key: "score" },
    { element: els.attributionModeCluster, key: "cluster" },
    { element: els.attributionModeContribution, key: "contrib" },
  ]
    .filter((mode) => mode.element.checked)
    .map((mode) => mode.key);
}

function setAttributionModes(modes) {
  const modeSet = new Set(modes);
  els.attributionModeScore.checked = modeSet.has("score");
  els.attributionModeCluster.checked = modeSet.has("cluster");
  els.attributionModeContribution.checked = modeSet.has("contrib");
}

function getDotViewMode() {
  if (els.toggleAttnDots.checked) {
    return VIEW_MODE_ATTN;
  }
  if (els.toggleBinaryDots.checked) {
    return VIEW_MODE_BINARY;
  }
  return VIEW_MODE_CONTINUOUS;
}

function setDotViewMode(mode) {
  if (mode === VIEW_MODE_ATTN) {
    els.toggleAttnDots.checked = true;
  } else if (mode === VIEW_MODE_BINARY) {
    els.toggleBinaryDots.checked = true;
  } else {
    els.dotModeContinuous.checked = true;
  }
}

function applyUiStateFromQuery() {
  const { attributionModes, dotViewMode, patch, requestedOverlays } =
    SilicaSlideViewerQuery.parseInitialState({
      search: window.location.search,
      imageLayerTileSources: IMAGE_LAYER_TILE_SOURCES,
    });
  if (requestedOverlays !== null) {
    for (const overlay of getOverlayToggleConfig()) {
      overlay.element.checked = requestedOverlays.has(overlay.key);
    }
  }
  if (attributionModes !== null) {
    if (requestedOverlays === null) {
      els.toggleAttributionText.checked = true;
    }
    setAttributionModes(attributionModes);
  }
  if (dotViewMode !== null) {
    setDotViewMode(dotViewMode);
  }
  commit(patch);
}

function syncUiStateQuery(slideKey = state.pendingSlideKey || state.currentSlideKey) {
  const url = new URL(window.location.href);
  const enabledOverlays = getOverlayToggleConfig()
    .filter((overlay) => overlay.element.checked)
    .map((overlay) => overlay.key);
  const dotViewMode = getDotViewMode();
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
    clampNumber(state.partitionFillAlpha, 0, 1) * 100,
  );
  const binaryThresholdValue = Math.round(
    clampNumber(state.binaryDotThreshold, 0, 1) * 100,
  );
  const attnScaleMaxValue =
    state.attnScoreMin === null ||
    state.attnScoreMax === null ||
    state.attnScoreDataMax === null
      ? null
      : clampNumber(state.attnScoreMax, state.attnScoreMin, state.attnScoreDataMax);

  const viewportState = captureViewportState();
  url.search = SilicaSlideViewerQuery.buildSearchParams({
    activeImageLayer: state.activeImageLayer,
    attnScaleMaxValue,
    attributionModes: getAttributionModes(),
    binaryThresholdValue,
    boxSizeValue,
    clusterQueryValue: SilicaSlideViewerQuery.formatClusterQueryValue({
      activeClusterFilters: state.activeClusterFilters,
      allClusterIds: state.cells ? getAllClusterIds() : [],
      hasCells: Boolean(state.cells),
      pendingClusterFilterIds: state.pendingClusterFilterIds,
    }),
    currentExperiment: state.currentExperiment,
    dotSizeValue,
    dotViewMode,
    enabledOverlays,
    partitionAlphaValue,
    slideKey,
    viewportState,
  }).toString();

  window.history.replaceState({}, "", url);
}

function scheduleViewportQuerySync() {
  SilicaSlideViewport.scheduleViewportQuerySync({
    state,
    syncUiStateQuery,
  });
}

function drawCellOverlay(context, visible) {
  drawCellOverlayFromState({
    context,
    visible,
    state,
    els,
    getAttributionModes,
    applyPendingDotRadiusValue,
    applyPendingBoxSizeValue,
    getRenderedDotRadius,
    updateDotSizeControl,
    updateBoxSizeControl,
  });
}

function updateViewportSummary(visible, viewportCells = visible) {
  const visibleCount = viewportCells.indices.length;
  els.visibleCellCount.textContent = formatInteger(visibleCount);

  if (!visibleCount) {
    els.meanTumorScore.textContent = "-";
    els.meanTumorScore.style.color = "";
  } else {
    const meanTumorScore = meanTumorScoreForCellIndices(
      state.cells,
      viewportCells.indices,
    );
    setPercentMetric("meanTumorScore", meanTumorScore);
  }

  const topCluster = getTopClusterForCells(state.cells, viewportCells.indices);
  if (!topCluster) {
    els.topCluster.textContent = "-";
    return;
  }
  els.topCluster.textContent = `C${topCluster.cluster} (${topCluster.count})`;
}

function drawScoreHistogram(indices) {
  drawScoreHistogramChart({
    canvas: els.scoreHistogram,
    cells: state.cells,
    indices,
  });
}

function drawClusterHistogram(indices) {
  drawClusterHistogramChart({
    canvas: els.clusterHistogram,
    cells: state.cells,
    indices,
    numClusters: state.manifest.num_clusters,
  });
}

function dziAssetUrl(relativePath) {
  return slideDziAssetUrl(state.currentSlideKey, relativePath);
}

function slideDziAssetUrl(slideKey, relativePath) {
  return SilicaSlideAssets.dziAssetUrl({
    relativePath,
    slideKey,
  });
}
