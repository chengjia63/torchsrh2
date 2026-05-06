const SilicaSlideControls = (() => {
  const {
    BOX_SIZE_IMAGE_MAX,
    BOX_SIZE_IMAGE_MIN,
    DEFAULT_BOX_SIZE_IMAGE_PX,
    DEFAULT_DOT_DIAMETER_IMAGE_PX,
    DOT_DIAMETER_IMAGE_MAX,
    DOT_DIAMETER_IMAGE_MIN,
    FILTER_ALL_VALUE,
    SIDEBAR_WIDTH_MAX_PERCENT,
    SIDEBAR_WIDTH_MIN_PERCENT,
    SIDEBAR_WIDTH_MIN_PX,
  } = SilicaSiteConfig;
  const { clampNumber } = SilicaSiteUtils;

  function bindControls({ state, els, handlers }) {
    bindTopbarSelectionReveal(state);
    bindResponsiveExperimentPicker(els);
    bindResponsiveTopbarPickerControls(els);
    bindSidebarCollapse({ state, onChange: handlers.refreshViewerAfterLayoutResize });

    els.diagnosisSelect.addEventListener("change", (event) => {
      handlers.revealTopbarSelectionLabels();
      handlers.applyFilters({
        diagnosis:
          event.target.value === FILTER_ALL_VALUE ? "" : event.target.value,
      });
    });

    els.infiltrationSelect.addEventListener("change", (event) => {
      handlers.revealTopbarSelectionLabels();
      handlers.applyFilters({
        infiltration:
          event.target.value === FILTER_ALL_VALUE ? "" : event.target.value,
      });
    });

    els.slideSelect.addEventListener("change", (event) => {
      handlers.revealTopbarSelectionLabels();
      if (event.target.value) {
        handlers.syncUiStateQuery(event.target.value);
        handlers.loadSlideFromUi(event.target.value, {
          syncEmptyFilters: true,
          preserveViewport: true,
        });
      }
    });
    els.previousSlideButton.addEventListener("click", () => {
      handlers.navigateFilteredSlides(-1);
    });
    els.nextSlideButton.addEventListener("click", () => {
      handlers.navigateFilteredSlides(1);
    });
    els.experimentSelect.addEventListener("change", (event) => {
      const requestedExperiment = String(event.target.value ?? "").trim();
      if (!requestedExperiment) {
        return;
      }
      if (
        !handlers.getSlideAvailableExperiments(state.currentSlideKey).includes(
          requestedExperiment,
        )
      ) {
        handlers.populateExperimentSelector(state.currentSlideKey);
        return;
      }
      handlers.commit({ currentExperiment: requestedExperiment }, { query: true });
      if (state.currentSlideKey) {
        handlers.loadSlideFromUi(state.currentSlideKey);
      }
    });
    document.querySelector(".image-layer-pill").addEventListener("click", (event) => {
      const button = event.target.closest(".image-layer-button");
      if (!button) {
        return;
      }
      handlers.setActiveImageLayer(button.dataset.imageLayer);
    });
    bindLayoutResizeHandle({
      state,
      updateFromPointer(clientX) {
        updateSidebarWidthFromPointer({
          clientX,
          onResize: handlers.refreshViewerAfterLayoutResize,
        });
      },
      onEnd: handlers.refreshViewerAfterLayoutResize,
    });
    els.clusterFilterInput.addEventListener("change", () => {
      handlers.applyClusterFilterInputValue();
    });
    els.clusterFilterInput.addEventListener("blur", () => {
      handlers.applyClusterFilterInputValue();
    });
    els.clusterFilterInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      handlers.applyClusterFilterInputValue();
    });

    bindOverlayCheckboxes({
      els,
      onChange(input) {
        normalizeOverlayCheckboxes(els, input);
        handlers.syncOverlayControlVisibility();
        handlers.commit({}, { query: true, redraw: true });
      },
    });
    els.dotSizeSlider.addEventListener("input", (event) => {
      const sliderValue = Number(event.target.value);
      if (!Number.isFinite(sliderValue)) {
        return;
      }
      const clampedSliderValue = clampNumber(
        sliderValue,
        DOT_DIAMETER_IMAGE_MIN,
        DOT_DIAMETER_IMAGE_MAX,
      );
      handlers.commit(
        {
          dotDiameterImagePx: clampedSliderValue,
          pendingDotRadiusValue: null,
        },
        { query: true, redraw: true },
      );
      handlers.updateDotSizeControl(clampedSliderValue);
    });
    els.boxSizeSlider.addEventListener("input", (event) => {
      const sliderValue = Number(event.target.value);
      if (!Number.isFinite(sliderValue)) {
        return;
      }
      const clampedSliderValue = clampNumber(
        sliderValue,
        BOX_SIZE_IMAGE_MIN,
        BOX_SIZE_IMAGE_MAX,
      );
      handlers.commit(
        {
          boxSizeImagePx: clampedSliderValue,
          pendingBoxSizeValue: null,
        },
        { query: true, redraw: true },
      );
      handlers.updateBoxSizeControl(clampedSliderValue);
    });
    els.partitionAlphaSlider.addEventListener("input", (event) => {
      const sliderValue = Number(event.target.value);
      if (!Number.isFinite(sliderValue)) {
        return;
      }
      const partitionFillAlpha = clampNumber(sliderValue / 100, 0, 1);
      handlers.commit({ partitionFillAlpha }, { query: true, redraw: true });
      handlers.updatePartitionAlphaControl(state.partitionFillAlpha);
    });
    els.binaryThresholdSlider.addEventListener("input", (event) => {
      const sliderValue = Number(event.target.value);
      if (!Number.isFinite(sliderValue)) {
        return;
      }
      const binaryDotThreshold = clampNumber(sliderValue / 100, 0, 1);
      handlers.commit({ binaryDotThreshold }, { query: true, redraw: true });
      handlers.updateBinaryThresholdControl(state.binaryDotThreshold);
    });
    els.attnScaleMaxSlider.addEventListener("input", (event) => {
      const sliderValue = Number(event.target.value);
      if (!Number.isFinite(sliderValue)) {
        throw new Error(`Invalid attention scale max slider value: ${event.target.value}`);
      }
      const attnScoreMax = handlers.getAttnScaleMaxFromSliderValue(sliderValue);
      handlers.commit(
        {
          attnScoreMax,
          pendingAttnScaleMaxValue: null,
        },
        { query: true, redraw: true },
      );
      handlers.updateAttnScaleControl();
    });
    els.viewportBoundsForm.addEventListener("submit", (event) => {
      event.preventDefault();
    });
    for (const input of [els.viewportInputVx, els.viewportInputVy, els.viewportInputVz]) {
      input.addEventListener("input", () => {
        handlers.scheduleViewportAutoApply();
      });
      input.addEventListener("change", () => {
        handlers.scheduleViewportAutoApply({ immediate: true });
      });
    }

    handlers.syncOverlayControlVisibility();
    handlers.syncClusterFilterEditor();
    handlers.updateDotSizeControl(state.pendingDotRadiusValue ?? DEFAULT_DOT_DIAMETER_IMAGE_PX);
    handlers.updateBoxSizeControl(state.pendingBoxSizeValue ?? DEFAULT_BOX_SIZE_IMAGE_PX);
    handlers.updatePartitionAlphaControl(state.partitionFillAlpha);
    handlers.updateBinaryThresholdControl(state.binaryDotThreshold);
    window.addEventListener("resize", handlers.scheduleRedraw);
    window.addEventListener("resize", () => {
      handlers.updateViewerViewportMargins();
      handlers.scheduleSidebarRedraw();
    });
  }

  function bindOverlayCheckboxes({ els, onChange }) {
    const redrawCheckboxes = [
      els.toggleDots,
      els.dotModeContinuous,
      els.toggleAttnDots,
      els.toggleBoxes,
      els.togglePartitionFill,
      els.toggleBinaryDots,
      els.toggleAttributionText,
      els.attributionModeScore,
      els.attributionModeCluster,
      els.attributionModeContribution,
    ];

    for (const input of redrawCheckboxes) {
      input.addEventListener("change", () => {
        onChange(input);
      });
    }
  }

  function normalizeOverlayCheckboxes(els, changedInput = null) {
    if (!els.toggleDots.checked) {
      return;
    }
    if (changedInput === els.dotModeContinuous) {
      els.toggleBinaryDots.checked = false;
      els.toggleAttnDots.checked = false;
      return;
    }
    if (changedInput === els.toggleBinaryDots && els.toggleBinaryDots.checked) {
      els.toggleAttnDots.checked = false;
      return;
    }
    if (changedInput === els.toggleAttnDots && els.toggleAttnDots.checked) {
      els.toggleBinaryDots.checked = false;
      return;
    }
    if (!els.toggleBinaryDots.checked && !els.toggleAttnDots.checked) {
      els.dotModeContinuous.checked = true;
    }
  }

  function bindTopbarSelectionReveal(state) {
    SilicaUI.bindTopbarSelectionReveal({
      state,
      triggerSelectors: [".topbar-filter-pill"],
    });
  }

  function bindResponsiveExperimentPicker(els) {
    SilicaUI.bindResponsivePlacement({
      element: document.querySelector(".topbar-experiment-picker"),
      mobileSlot: els.mobileExperimentPickerSlot,
      desktopParent: document.querySelector(".topbar-inner"),
      mediaQueryText: "(max-width: 1024px)",
      anchorLabel: "experiment picker desktop anchor",
    });
  }

  function bindResponsiveTopbarPickerControls(els) {
    const desktopParent = document.querySelector(".topbar-picker");
    const responsiveControls = [
      {
        element: document.querySelector(".topbar-filter-pill"),
        mobileSlot: els.mobileFilterPickerSlot,
        anchorLabel: "filter picker desktop anchor",
      },
      {
        element: document.querySelector(".topbar-slide-nav-pill"),
        mobileSlot: els.narrowMobileSlideNavSlot,
        anchorLabel: "slide navigation desktop anchor",
      },
    ];
    for (const responsiveControl of responsiveControls) {
      SilicaUI.bindResponsivePlacement({
        ...responsiveControl,
        desktopParent,
        mediaQueryText: "(max-width: 700px)",
      });
    }
  }

  function bindSidebarCollapse({ state, onChange }) {
    SilicaUI.bindSidebarCollapse({
      state,
      buttonIds: ["sidebarCollapseButton", "sidebarHeaderCollapseButton"],
      onChange,
    });
  }

  function bindLayoutResizeHandle({ state, updateFromPointer, onEnd }) {
    SilicaUI.bindLayoutResize({
      state,
      handleId: "layoutResizeHandle",
      updateFromPointer,
      onEnd,
    });
  }

  function updateSidebarWidthFromPointer({ clientX, onResize }) {
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
    const rawPercent = ((layoutRect.right - clientX) / layoutRect.width) * 100;
    const sidebarWidthPercent = clampNumber(
      rawPercent,
      minPercent,
      SIDEBAR_WIDTH_MAX_PERCENT,
    );
    document.documentElement.style.setProperty(
      "--sidebar-width",
      `${sidebarWidthPercent.toFixed(2)}%`,
    );
    onResize();
  }

  return Object.freeze({
    bindControls,
    normalizeOverlayCheckboxes,
  });
})();
