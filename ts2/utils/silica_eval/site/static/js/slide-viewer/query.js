const SilicaSlideViewerQuery = (() => {
  const {
    ATTN_SCALE_MAX_QUERY_PARAM,
    ATTRIBUTION_QUERY_PARAM,
    BINARY_THRESHOLD_QUERY_PARAM,
    BOX_SIZE_IMAGE_MAX,
    BOX_SIZE_IMAGE_MIN,
    BOX_SIZE_QUERY_PARAM,
    CLUSTER_QUERY_PARAM,
    DEFAULT_BINARY_DOT_SCORE_THRESHOLD,
    DEFAULT_BOX_SIZE_IMAGE_PX,
    DEFAULT_DOT_DIAMETER_IMAGE_PX,
    DEFAULT_PARTITION_FILL_ALPHA,
    DOT_DIAMETER_IMAGE_MAX,
    DOT_DIAMETER_IMAGE_MIN,
    DOT_SIZE_QUERY_PARAM,
    EXPERIMENT_QUERY_PARAM,
    IMAGE_LAYER_QUERY_PARAM,
    OVERLAY_QUERY_PARAM,
    PARTITION_ALPHA_QUERY_PARAM,
    SLIDE_QUERY_PARAM,
    VIEW_MODE_ATTN,
    VIEW_MODE_BINARY,
    VIEW_MODE_CONTINUOUS,
    VIEW_QUERY_PARAM,
    VIEWPORT_X_QUERY_PARAM,
    VIEWPORT_Y_QUERY_PARAM,
    VIEWPORT_ZOOM_QUERY_PARAM,
  } = SilicaSiteConfig;
  const { clampNumber } = SilicaSiteUtils;
  const ATTRIBUTION_MODES = Object.freeze(["score", "cluster", "contrib"]);
  const ATTRIBUTION_NONE_VALUE = "none";

  function parseInitialState({ search, imageLayerTileSources }) {
    const searchParams = new URLSearchParams(search);
    const patch = {
      pendingDotRadiusValue: DEFAULT_DOT_DIAMETER_IMAGE_PX,
      pendingBoxSizeValue: DEFAULT_BOX_SIZE_IMAGE_PX,
      partitionFillAlpha: DEFAULT_PARTITION_FILL_ALPHA,
      binaryDotThreshold: DEFAULT_BINARY_DOT_SCORE_THRESHOLD,
    };

    const requestedExperiment = (searchParams.get(EXPERIMENT_QUERY_PARAM) ?? "").trim();
    if (requestedExperiment) {
      patch.pendingExperiment = requestedExperiment;
    }

    const requestedOverlays = parseOverlaySet(searchParams.get(OVERLAY_QUERY_PARAM));
    const attributionModes = resolveAttributionModes(
      searchParams.get(ATTRIBUTION_QUERY_PARAM),
      requestedOverlays,
    );
    const dotViewMode = resolveDotViewMode(
      searchParams.get(VIEW_QUERY_PARAM),
      requestedOverlays,
    );

    const dotSizeRaw = parseNumericParam(searchParams, DOT_SIZE_QUERY_PARAM);
    if (Number.isFinite(dotSizeRaw)) {
      patch.pendingDotRadiusValue = clampNumber(
        dotSizeRaw,
        DOT_DIAMETER_IMAGE_MIN,
        DOT_DIAMETER_IMAGE_MAX,
      );
    }

    const boxSizeRaw = parseNumericParam(searchParams, BOX_SIZE_QUERY_PARAM);
    if (Number.isFinite(boxSizeRaw)) {
      patch.pendingBoxSizeValue = clampNumber(
        boxSizeRaw,
        BOX_SIZE_IMAGE_MIN,
        BOX_SIZE_IMAGE_MAX,
      );
    }

    const alphaRaw = parseNumericParam(searchParams, PARTITION_ALPHA_QUERY_PARAM);
    if (Number.isFinite(alphaRaw)) {
      patch.partitionFillAlpha = clampNumber(alphaRaw / 100, 0, 1);
    }

    const thresholdRaw = parseNumericParam(searchParams, BINARY_THRESHOLD_QUERY_PARAM);
    if (Number.isFinite(thresholdRaw)) {
      patch.binaryDotThreshold = clampNumber(thresholdRaw / 100, 0, 1);
    }

    const attnScaleMaxRaw = parseNumericParam(searchParams, ATTN_SCALE_MAX_QUERY_PARAM);
    if (Number.isFinite(attnScaleMaxRaw)) {
      patch.pendingAttnScaleMaxValue = attnScaleMaxRaw;
    }

    const clustersRaw = searchParams.get(CLUSTER_QUERY_PARAM);
    if (clustersRaw === "all" || clustersRaw === "none") {
      patch.pendingClusterFilterIds = clustersRaw;
    } else if (clustersRaw) {
      patch.pendingClusterFilterIds = clustersRaw
        .split(",")
        .map((value) => Number.parseInt(value, 10))
        .filter((value) => Number.isInteger(value));
    }

    const imageLayerRaw = searchParams.get(IMAGE_LAYER_QUERY_PARAM);
    if (Object.prototype.hasOwnProperty.call(imageLayerTileSources, imageLayerRaw)) {
      patch.activeImageLayer = imageLayerRaw;
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
      patch.pendingViewportState = {
        center: { x: viewportXRaw, y: viewportYRaw },
        zoom: viewportZoomRaw,
      };
    }

    return {
      attributionModes,
      dotViewMode,
      patch,
      requestedOverlays,
    };
  }

  function buildSearchParams({
    activeImageLayer,
    attnScaleMaxValue,
    attributionModes,
    binaryThresholdValue,
    boxSizeValue,
    clusterQueryValue,
    currentExperiment,
    dotSizeValue,
    dotViewMode,
    enabledOverlays,
    partitionAlphaValue,
    slideKey,
    viewportState,
  }) {
    const nextSearchParams = new URLSearchParams();
    if (slideKey) {
      nextSearchParams.set(SLIDE_QUERY_PARAM, slideKey);
    }
    if (currentExperiment) {
      nextSearchParams.set(EXPERIMENT_QUERY_PARAM, currentExperiment);
    }
    nextSearchParams.set(IMAGE_LAYER_QUERY_PARAM, activeImageLayer);
    nextSearchParams.set(OVERLAY_QUERY_PARAM, enabledOverlays.join(","));
    nextSearchParams.set(DOT_SIZE_QUERY_PARAM, dotSizeValue.toFixed(1));
    nextSearchParams.set(BOX_SIZE_QUERY_PARAM, boxSizeValue.toFixed(1));
    nextSearchParams.set(PARTITION_ALPHA_QUERY_PARAM, `${partitionAlphaValue}`);
    nextSearchParams.set(
      ATTRIBUTION_QUERY_PARAM,
      attributionModes.length > 0 ? attributionModes.join(",") : ATTRIBUTION_NONE_VALUE,
    );
    if (viewportState) {
      nextSearchParams.set(VIEWPORT_X_QUERY_PARAM, `${viewportState.center.x.toFixed(5)}`);
      nextSearchParams.set(VIEWPORT_Y_QUERY_PARAM, `${viewportState.center.y.toFixed(5)}`);
      nextSearchParams.set(VIEWPORT_ZOOM_QUERY_PARAM, `${viewportState.zoom.toFixed(5)}`);
    }
    nextSearchParams.set(VIEW_QUERY_PARAM, dotViewMode);
    nextSearchParams.set(BINARY_THRESHOLD_QUERY_PARAM, `${binaryThresholdValue}`);
    if (attnScaleMaxValue !== null) {
      nextSearchParams.set(ATTN_SCALE_MAX_QUERY_PARAM, attnScaleMaxValue.toPrecision(6));
    }
    nextSearchParams.set(CLUSTER_QUERY_PARAM, clusterQueryValue);
    return nextSearchParams;
  }

  function formatClusterQueryValue({
    activeClusterFilters,
    allClusterIds,
    hasCells,
    pendingClusterFilterIds,
  }) {
    if (hasCells) {
      if (activeClusterFilters.size === 0) {
        return "none";
      }
      if (
        allClusterIds.length > 0 &&
        allClusterIds.every((clusterId) => activeClusterFilters.has(clusterId))
      ) {
        return "all";
      }
      return Array.from(activeClusterFilters)
        .sort((left, right) => left - right)
        .join(",");
    }
    if (Array.isArray(pendingClusterFilterIds)) {
      return [...pendingClusterFilterIds]
        .sort((left, right) => left - right)
        .join(",");
    }
    if (pendingClusterFilterIds === "none") {
      return "none";
    }
    return "all";
  }

  function parseNumericParam(searchParams, queryParam) {
    const rawValue = searchParams.get(queryParam);
    return rawValue === null ? Number.NaN : Number(rawValue);
  }

  function parseOverlaySet(rawValue) {
    if (rawValue === null) {
      return null;
    }
    return new Set(
      rawValue
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
  }

  function resolveAttributionModes(rawValue, requestedOverlays) {
    const attributionRaw = (rawValue ?? "").trim();
    if (attributionRaw === ATTRIBUTION_NONE_VALUE) {
      return [];
    }
    const attributionModes = attributionRaw
      .split(",")
      .map((value) => value.trim())
      .filter((value) => ATTRIBUTION_MODES.includes(value));
    if (attributionModes.length > 0) {
      return ATTRIBUTION_MODES.filter((mode) => attributionModes.includes(mode));
    }
    if (!requestedOverlays) {
      return null;
    }
    const legacyModes = ATTRIBUTION_MODES.filter((mode) => requestedOverlays.has(mode));
    return legacyModes.length > 0 ? legacyModes : null;
  }

  function resolveDotViewMode(rawValue, requestedOverlays) {
    const viewRaw = (rawValue ?? "").trim();
    if (viewRaw === VIEW_MODE_ATTN) {
      return VIEW_MODE_ATTN;
    }
    if (viewRaw === VIEW_MODE_BINARY) {
      return VIEW_MODE_BINARY;
    }
    if (viewRaw === VIEW_MODE_CONTINUOUS) {
      return VIEW_MODE_CONTINUOUS;
    }
    if (requestedOverlays?.has(VIEW_MODE_ATTN)) {
      return VIEW_MODE_ATTN;
    }
    return null;
  }

  return Object.freeze({
    buildSearchParams,
    formatClusterQueryValue,
    parseInitialState,
  });
})();
