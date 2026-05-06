const SilicaSlideOverlayRenderer = (() => {
  const {
    BINARY_DOT_STROKE_STYLE,
    CLUSTER_LABEL_VIEW_WIDTH_RATIO,
    CONTRIBUTION_LABEL_VIEW_WIDTH_RATIO,
    DEFAULT_BOX_SIZE_IMAGE_PX,
    DEFAULT_DOT_DIAMETER_IMAGE_PX,
    MIN_SCREEN_BOX_SIZE_PX,
    OVERLAY_BOX_STROKE_STYLE,
    OVERLAY_TEXT_FILL_STYLE,
    OVERLAY_TEXT_STROKE_STYLE,
    PARTITION_BUCKET_SIZE_PX,
    PARTITION_HIGH_RES_VIEW_WIDTH_RATIO,
    PARTITION_SAMPLE_STEP_PX,
    PARTITION_VIEW_MARGIN_PX,
    PARTITION_ZOOMED_OUT_SAMPLE_STEP_PX,
    SCORE_LABEL_VIEW_WIDTH_RATIO,
    TEXT_RENDER_LIMIT,
  } = SilicaSiteConfig;

  const {
    buildPartitionBuckets,
    ensureMinimumScreenRectSize,
    expandImageRectForScreenMargin,
    findNearestPartitionPoint,
  } = SilicaSlideGeometry;

  const {
    getAttnDotColor,
    getDotOverlayColor,
  } = SilicaSlideColors;

  function drawCellOverlay({
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
  }) {
    const showDots = els.toggleDots.checked;
    const showBoxes = els.toggleBoxes.checked;
    const showPartitionFill = els.togglePartitionFill.checked;
    const attributionModes = new Set(getAttributionModes());
    const viewWidthRatio = visible.bounds.width / state.manifest.image_width;

    const renderScoreText =
      els.toggleAttributionText.checked &&
      attributionModes.has("score") &&
      visible.indices.length <= TEXT_RENDER_LIMIT &&
      viewWidthRatio <= SCORE_LABEL_VIEW_WIDTH_RATIO;
    const renderClusterText =
      els.toggleAttributionText.checked &&
      attributionModes.has("cluster") &&
      visible.indices.length <= TEXT_RENDER_LIMIT &&
      viewWidthRatio <= CLUSTER_LABEL_VIEW_WIDTH_RATIO;
    const renderContributionText =
      els.toggleAttributionText.checked &&
      attributionModes.has("contrib") &&
      visible.indices.length <= TEXT_RENDER_LIMIT &&
      viewWidthRatio <= CONTRIBUTION_LABEL_VIEW_WIDTH_RATIO;
    applyPendingDotRadiusValue();
    applyPendingBoxSizeValue();
    const dotRadius = getRenderedDotRadius();
    const showAttnDots =
      showDots &&
      els.toggleAttnDots.checked &&
      Array.isArray(state.cells?.attn_score) &&
      state.cells.attn_score.length > 0;
    const showBinaryDots = showDots && els.toggleBinaryDots.checked && !showAttnDots;
    const boxSizeImagePx =
      state.pendingBoxSizeValue ?? state.boxSizeImagePx ?? DEFAULT_BOX_SIZE_IMAGE_PX;
    const dotStrokeWidth = 1.6;
    updateDotSizeControl(
      state.pendingDotRadiusValue ?? state.dotDiameterImagePx ?? DEFAULT_DOT_DIAMETER_IMAGE_PX,
    );
    updateBoxSizeControl(boxSizeImagePx);

    if (showPartitionFill) {
      drawPartitionOverlay(context, state, els, viewWidthRatio, visible.bounds, {
        useBinaryColors: showBinaryDots,
      });
    }

    context.textAlign = "center";
    context.textBaseline = "middle";
    context.font = "11px 'Roboto Mono', monospace";

    for (const index of visible.indices) {
      const point = imageToScreenPoint(state, state.cells.x[index], state.cells.y[index]);
      if (!point) {
        continue;
      }

      if (showDots && !showAttnDots) {
        const dotColor = getDotOverlayColor({
          cells: state.cells,
          index,
          useBinaryColors: showBinaryDots,
          binaryDotThreshold: state.binaryDotThreshold,
        });
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

      if (showAttnDots) {
        const attnColor = getAttnDotColor({
          cells: state.cells,
          index,
          attnScoreMin: state.attnScoreMin,
          attnScoreMax: state.attnScoreMax,
        });
        context.save();
        context.globalAlpha = 0.5;
        context.beginPath();
        context.fillStyle = attnColor;
        context.arc(point.x, point.y, dotRadius, 0, Math.PI * 2);
        context.fill();
        context.restore();
        context.lineWidth = dotStrokeWidth;
        context.strokeStyle = attnColor;
        context.stroke();
      }

      if (showBoxes) {
        const boxRect = imageRectToScreenRect(
          state,
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
      if (textLines.length === 0) {
        continue;
      }

      const lineHeight = 12;
      const firstTextY = point.y - 12 - ((textLines.length - 1) * lineHeight) / 2;
      for (const [lineIndex, textLine] of textLines.entries()) {
        const textY = firstTextY + lineIndex * lineHeight;
        context.strokeStyle = OVERLAY_TEXT_STROKE_STYLE;
        context.lineWidth = 3;
        context.strokeText(textLine, point.x, textY);
        context.fillStyle = OVERLAY_TEXT_FILL_STYLE;
        context.fillText(textLine, point.x, textY);
      }
    }
  }

  function drawPartitionOverlay(context, state, els, viewWidthRatio, visibleImageBounds, options = {}) {
    const { useBinaryColors = false } = options;
    const overlayWidth = els.overlayCanvas?.getBoundingClientRect().width ?? 0;
    const overlayHeight = els.overlayCanvas?.getBoundingClientRect().height ?? 0;
    if (overlayWidth <= 0 || overlayHeight <= 0) {
      return;
    }

    const candidatePoints = getPartitionCandidateScreenPoints(
      state,
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
        const sampleImagePoint = screenToImagePoint(state, sampleX, sampleY);
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
    state,
    visibleImageBounds,
    overlayWidth,
    overlayHeight,
    options = {},
  ) {
    const { useBinaryColors = false } = options;
    const expandedImageBounds = expandImageRectForScreenMargin(
      {
        imageHeight: state.manifest.image_height,
        imageRect: visibleImageBounds,
        imageWidth: state.manifest.image_width,
        overlayHeight,
        overlayWidth,
        screenMarginPx: PARTITION_VIEW_MARGIN_PX,
      },
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
      const point = imageToScreenPoint(state, state.cells.x[index], state.cells.y[index]);
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
        color: getDotOverlayColor({
          cells: state.cells,
          index,
          useBinaryColors,
          binaryDotThreshold: state.binaryDotThreshold,
        }),
      });
    }
    return points;
  }

  function imageRectToScreenRect(state, centerX, centerY, width, height) {
    const halfWidth = width / 2;
    const halfHeight = height / 2;
    const topLeft = imageToScreenPoint(state, centerX - halfWidth, centerY - halfHeight);
    const bottomRight = imageToScreenPoint(state, centerX + halfWidth, centerY + halfHeight);
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

  function screenToImagePoint(state, x, y) {
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

  function imageToScreenPoint(state, x, y) {
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

  return Object.freeze({
    drawCellOverlay,
  });
})();
