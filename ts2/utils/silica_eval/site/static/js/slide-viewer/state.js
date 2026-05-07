const SilicaSlideState = (() => {
  const {
    DEFAULT_BINARY_DOT_SCORE_THRESHOLD,
    DEFAULT_PARTITION_FILL_ALPHA,
  } = SilicaSiteConfig;

  function createInitialState() {
    return {
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
      pendingSlideKey: null,
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
      partitionFillAlpha: DEFAULT_PARTITION_FILL_ALPHA,
      binaryDotThreshold: DEFAULT_BINARY_DOT_SCORE_THRESHOLD,
      attnScoreMin: null,
      attnScoreMax: null,
      attnScoreDataMax: null,
      pendingAttnScaleMaxValue: null,
      activeClusterFilters: new Set(),
      topbarSelectionRevealed: false,
      activeImageLayer: "srhvhe",
      pendingImageLayerFallback: null,
      layoutResizeActive: false,
      sidebarCollapsed: false,
    };
  }

  return Object.freeze({
    createInitialState,
  });
})();
