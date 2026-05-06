const SilicaSlideCharts = (() => {
  const {
    buildClusterXAxisTickIndices,
    buildHistogram,
    drawBarChart,
    resizeCanvas,
  } = SilicaSiteUtils;
  const {
    CLUSTER_HISTOGRAM_FILL_STYLE,
    SCORE_HISTOGRAM_FILL_STYLE,
  } = SilicaSiteConfig;

  function drawScoreHistogram({ canvas, cells, indices }) {
    const { width, height } = resizeCanvas(canvas);
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    const histogram = buildHistogram(
      indices.map((index) => cells.tumor_score[index]),
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

  function drawClusterHistogram({ canvas, cells, indices, numClusters }) {
    const { width, height } = resizeCanvas(canvas);
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    const clusterCounts = Array.from({ length: numClusters }, () => 0);
    for (const index of indices) {
      clusterCounts[cells.dominant_cluster[index]] += 1;
    }
    drawBarChart({
      context,
      width,
      height,
      xTickLabels: clusterCounts.map((_, index) => `${index}`),
      xTickIndices: buildClusterXAxisTickIndices(clusterCounts.length),
      values: clusterCounts,
      fillStyle: CLUSTER_HISTOGRAM_FILL_STYLE,
      emptyLabel: "No visible cells",
    });
  }

  return Object.freeze({
    drawClusterHistogram,
    drawScoreHistogram,
  });
})();
