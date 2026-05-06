const SilicaSlideColors = (() => {
  const {
    ATTN_SCALE_MIN,
    PLASMA_COLOR_STOPS,
    TUMOR_SCORE_COLOR_STOPS,
  } = SilicaSiteConfig;
  const { colorFromStops } = SilicaSiteUtils;

  function getTumorScoreColor(tumorScore) {
    return colorFromStops(tumorScore, TUMOR_SCORE_COLOR_STOPS);
  }

  function getDotOverlayColor({
    cells,
    index,
    useBinaryColors,
    binaryDotThreshold,
  }) {
    if (!useBinaryColors) {
      return cells.dot_color[index];
    }
    const tumorScore = Number(cells.tumor_score[index]);
    const binaryScore =
      Number.isFinite(tumorScore) && tumorScore >= binaryDotThreshold ? 1 : 0;
    return getTumorScoreColor(binaryScore);
  }

  function getAttnDotColor({
    cells,
    index,
    attnScoreMin,
    attnScoreMax,
  }) {
    const rawValue = Number(cells.attn_score[index]);
    if (!Number.isFinite(rawValue)) {
      throw new Error(`Invalid attention score at index ${index}: ${cells.attn_score[index]}`);
    }
    if (attnScoreMin === null || attnScoreMax === null) {
      throw new Error("Attention score range is unavailable");
    }
    const scoreRange = attnScoreMax - ATTN_SCALE_MIN;
    const normalizedValue = scoreRange === 0 ? 1 : (rawValue - ATTN_SCALE_MIN) / scoreRange;
    return colorFromStops(normalizedValue, PLASMA_COLOR_STOPS);
  }

  return Object.freeze({
    getAttnDotColor,
    getDotOverlayColor,
    getTumorScoreColor,
  });
})();
