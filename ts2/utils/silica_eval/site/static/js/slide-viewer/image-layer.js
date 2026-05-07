const SilicaSlideImageLayer = (() => {
  const { IMAGE_LAYER_TILE_SOURCES } = SilicaSiteConfig;

  function initializeViewer({
    state,
    commit,
    els,
    dziAssetUrl,
    handlers,
  }) {
    const initialTileSource = dziAssetUrl(IMAGE_LAYER_TILE_SOURCES[state.activeImageLayer]);
    const viewportMargins = SilicaSlideViewport.getViewerViewportMargins();
    const viewer = OpenSeadragon({
      id: "viewer",
      prefixUrl: window.SILICA_SLIDE_VIEWER.openSeadragonPrefix,
      tileSources: initialTileSource,
      showNavigator: true,
      navigatorPosition: "BOTTOM_LEFT",
      navigationControlAnchor: OpenSeadragon.ControlAnchor.BOTTOM_RIGHT,
      viewportMargins,
      visibilityRatio: 1,
      constrainDuringPan: true,
      animationTime: 0.7,
      blendTime: 0.1,
      minZoomLevel: 0.5,
      maxZoomPixelRatio: 2.5,
      zoomPerScroll: 1.25,
    });
    commit({ viewer });
    attachOverlayCanvasToViewer(els);

    state.viewer.addHandler("open", () => {
      commit({ pendingImageLayerFallback: null });
      handlers.setTopbarStatus("ready");
      handlers.restoreOrInitializeViewportState();
      handlers.applyPendingDotRadiusValue();
      handlers.syncUiStateQuery();
      handlers.scheduleRedraw();
      handlers.scheduleSidebarRedraw({ immediate: true });
    });
    state.viewer.addHandler("open-failed", (event) => {
      if (state.pendingImageLayerFallback !== null) {
        commit({
          activeImageLayer: state.pendingImageLayerFallback,
          pendingImageLayerFallback: null,
        });
        handlers.syncImageLayerControls();
        handlers.openActiveImageLayer();
        return;
      }
      const message = event?.message || event?.source || "unknown image source";
      handlers.setTopbarStatus("error");
      handlers.setViewportEditorStatus(`Failed to open DZI image: ${message}`);
    });
    state.viewer.addHandler("tile-load-failed", (event) => {
      const tileUrl = event?.tile?.url || event?.src || "unknown tile";
      handlers.setTopbarStatus("error");
      handlers.setViewportEditorStatus(`Failed to load DZI tile: ${tileUrl}`);
    });
    state.viewer.addHandler("animation", handlers.onViewportChanged);
    state.viewer.addHandler("resize", handlers.onViewportChanged);
    state.viewer.addHandler("pan", handlers.onViewportChanged);
    state.viewer.addHandler("zoom", handlers.onViewportChanged);
  }

  function attachOverlayCanvasToViewer(els) {
    const viewerContainer = document.querySelector("#viewer .openseadragon-container");
    if (!els.overlayCanvas || !viewerContainer) {
      throw new Error("Unable to attach overlay canvas to OpenSeadragon viewer");
    }
    viewerContainer.appendChild(els.overlayCanvas);
  }

  function setActiveImageLayer({
    state,
    commit,
    imageLayer,
    openActiveImageLayer,
    syncImageLayerControls,
  }) {
    if (!Object.prototype.hasOwnProperty.call(IMAGE_LAYER_TILE_SOURCES, imageLayer)) {
      throw new Error(`Unknown image layer: ${imageLayer}`);
    }
    if (state.activeImageLayer === imageLayer) {
      return;
    }
    commit({
      pendingImageLayerFallback: state.activeImageLayer,
      activeImageLayer: imageLayer,
    }, { query: true });
    syncImageLayerControls();
    openActiveImageLayer({ preserveViewport: true });
  }

  function syncImageLayerControls(state) {
    for (const button of document.querySelectorAll(".image-layer-button")) {
      const isActive = button.dataset.imageLayer === state.activeImageLayer;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-checked", isActive ? "true" : "false");
    }
  }

  function openActiveImageLayer({
    state,
    commit,
    slideDziAssetUrl,
    setTopbarStatus,
    captureViewportState,
    preserveViewport = false,
  }) {
    if (!state.viewer || !state.currentSlideKey) {
      return;
    }
    if (preserveViewport) {
      commit({ pendingViewportState: captureViewportState() });
    }
    const tileSource = slideDziAssetUrl(
      state.currentSlideKey,
      IMAGE_LAYER_TILE_SOURCES[state.activeImageLayer],
    );
    setTopbarStatus("loading");
    state.viewer.open(tileSource);
  }

  function refreshViewerAfterLayoutResize({
    state,
    scheduleRedraw,
    scheduleSidebarRedraw,
    scheduleViewportQuerySync,
  }) {
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

  return Object.freeze({
    initializeViewer,
    openActiveImageLayer,
    refreshViewerAfterLayoutResize,
    setActiveImageLayer,
    syncImageLayerControls,
  });
})();
