const SilicaSlideViewport = (() => {
  const {
    DEFAULT_VIEWPORT_CENTER_X,
    DEFAULT_VIEWPORT_CENTER_Y,
    DEFAULT_VIEWPORT_HEIGHT_RATIO,
    VIEWPORT_AUTO_APPLY_DELAY_MS,
  } = SilicaSiteConfig;
  const { clampNumber } = SilicaSiteUtils;

  function getViewerViewportMargins() {
    return {
      top: 0,
      right: 0,
      bottom: 0,
      left: 0,
    };
  }

  function updateViewerViewportMargins(state) {
    if (!state.viewer) {
      return;
    }
    state.viewer.viewportMargins = getViewerViewportMargins();
    state.viewer.viewport?.applyConstraints();
  }

  function captureViewportState(state) {
    if (!state.viewer || !state.viewer.viewport || !state.manifest) {
      return null;
    }
    const visibleImageRect = getVisibleImageRect(state);
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

  function restoreViewportState({ state, commit }) {
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
    const centerImageX = clampNumber(center.x, 0, 1) * state.manifest.image_width;
    const centerImageY = clampNumber(center.y, 0, 1) * state.manifest.image_height;
    const viewportRect = state.viewer.viewport.imageToViewportRectangle(
      centerImageX - imageWidth / 2,
      centerImageY - imageHeight / 2,
      imageWidth,
      imageHeight,
    );
    state.viewer.viewport.fitBounds(viewportRect, true);
    state.viewer.viewport.applyConstraints();
    commit({ pendingViewportState: null });
  }

  function restoreOrInitializeViewportState({ state, commit }) {
    if (state.pendingViewportState) {
      restoreViewportState({ state, commit });
      return;
    }
    commit({
      pendingViewportState: {
        center: {
          x: DEFAULT_VIEWPORT_CENTER_X,
          y: DEFAULT_VIEWPORT_CENTER_Y,
        },
        zoom: DEFAULT_VIEWPORT_HEIGHT_RATIO,
      },
    });
    restoreViewportState({ state, commit });
  }

  function collectVisibleCells({ state, applyClusterFilter = true }) {
    const imageRect = getVisibleImageRect(state);
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

  function getVisibleImageRect(state) {
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

  function getScreenPixelsPerImagePixel(state) {
    if (
      !state.viewer ||
      !state.viewer.viewport ||
      !state.manifest ||
      !state.viewer.world ||
      state.viewer.world.getItemCount() === 0
    ) {
      return null;
    }
    const visibleRect = getVisibleImageRect(state);
    const containerSize = state.viewer.viewport.getContainerSize();
    const containerHeight = Math.max(1, containerSize.y);
    const visibleHeight = Math.max(visibleRect.height, 1.0e-6);
    return containerHeight / visibleHeight;
  }

  function setViewportEditorStatus(els, message) {
    els.diagnosticsStatus.textContent = message || "";
  }

  function updateViewportInputs(els, bounds) {
    els.viewportInputVx.value = `${bounds.center.x.toFixed(2)}`;
    els.viewportInputVy.value = `${bounds.center.y.toFixed(2)}`;
    els.viewportInputVz.value = `${bounds.zoom.toFixed(2)}`;
    setViewportEditorStatus(els, "");
  }

  function applyViewportBoundsFromInputs({ state, els, commit, onApplied }) {
    if (!state.viewer?.viewport || !state.manifest) {
      return;
    }

    const rawVx = Number(els.viewportInputVx.value);
    const rawVy = Number(els.viewportInputVy.value);
    const rawVz = Number(els.viewportInputVz.value);
    if (
      !Number.isFinite(rawVx) ||
      !Number.isFinite(rawVy) ||
      !Number.isFinite(rawVz)
    ) {
      setViewportEditorStatus(
        els,
        "Enter numeric Center X, Center Y, and View Height values.",
      );
      return;
    }
    if (rawVz <= 0) {
      setViewportEditorStatus(els, "View Height must be greater than 0.");
      return;
    }

    commit({
      pendingViewportState: {
        center: {
          x: clampNumber(rawVx, 0, 1),
          y: clampNumber(rawVy, 0, 1),
        },
        zoom: clampNumber(rawVz, 1.0e-6, 1),
      },
    });
    restoreViewportState({ state, commit });
    updateViewportInputs(els, captureViewportState(state) || state.pendingViewportState);
    setViewportEditorStatus(els, "Viewport updated.");
    onApplied();
  }

  function scheduleViewportAutoApply({
    state,
    els,
    immediate = false,
    applyViewportBoundsFromInputs,
  }) {
    if (state.viewportApplyTimer !== null) {
      window.clearTimeout(state.viewportApplyTimer);
      state.viewportApplyTimer = null;
    }
    if (immediate) {
      applyViewportBoundsFromInputs();
      return;
    }
    setViewportEditorStatus(els, "Applying...");
    state.viewportApplyTimer = window.setTimeout(() => {
      state.viewportApplyTimer = null;
      applyViewportBoundsFromInputs();
    }, VIEWPORT_AUTO_APPLY_DELAY_MS);
  }

  function scheduleViewportQuerySync({ state, syncUiStateQuery }) {
    if (state.viewportQuerySyncTimer !== null) {
      window.clearTimeout(state.viewportQuerySyncTimer);
    }
    state.viewportQuerySyncTimer = window.setTimeout(() => {
      state.viewportQuerySyncTimer = null;
      syncUiStateQuery();
    }, 120);
  }

  return Object.freeze({
    applyViewportBoundsFromInputs,
    captureViewportState,
    collectVisibleCells,
    getScreenPixelsPerImagePixel,
    getViewerViewportMargins,
    getVisibleImageRect,
    restoreOrInitializeViewportState,
    restoreViewportState,
    scheduleViewportAutoApply,
    scheduleViewportQuerySync,
    setViewportEditorStatus,
    updateViewerViewportMargins,
    updateViewportInputs,
  });
})();
