const { fetchJson } = SilicaSiteUtils;

const predictionState = {
  charts: [],
  metrics: [],
  experiments: [],
  selectedChartId: "",
  selectedExperiment: "",
  selectedMetricsId: "",
  chartView: null,
  chartLoadToken: 0,
  yDomain: null,
  yDragStart: null,
  pendingChartRenderFrame: null,
  suppressPredictionClickUntil: 0,
  topbarStatus: "idle",
  topbarStatusHideTimer: null,
  sidebarCollapsed: false,
  layoutResizeActive: false,
  topbarSelectionRevealed: false,
};
const PREDICTION_CHART_Y_DOMAIN = [-5, 6];
const PREDICTION_Y_ZOOM_WHEEL_FACTOR = 1.0015;
const PREDICTION_Y_DRAG_THRESHOLD_PX = 4;
const PREDICTION_DRAG_CLICK_SUPPRESSION_MS = 250;
const EXPERIMENT_QUERY_PARAM = "experiment";
const Y_MIN_QUERY_PARAM = "y_min";
const Y_MAX_QUERY_PARAM = "y_max";
const SLIDE_QUERY_PARAM = "slide";

document.addEventListener("DOMContentLoaded", () => {
  void bootstrapPredictions().catch((error) => {
    console.error(error);
    setPredictionTopbarStatus("error");
    setPredictionStatus(String(error));
  });
});

async function bootstrapPredictions() {
  const slidePayload = await fetchJson(window.SILICA_PREDICTIONS.slidesUrl);
  predictionState.experiments = slidePayload.experiments;
  predictionState.selectedExperiment = resolveInitialPredictionExperiment(
    slidePayload.default_experiment,
  );
  predictionState.yDomain = resolveInitialPredictionYDomain();
  populatePredictionExperimentSelect(predictionState.experiments);
  bindPredictionControls();
  bindPredictionChartYAxisInteractions();
  bindPredictionTopbarSelectionReveal();
  bindPredictionSidebarControls();
  syncPredictionExperimentQuery();

  setPredictionTopbarStatus("loading");
  setPredictionStatus("Loading charts...");
  const [chartPayload, metricsPayload] = await Promise.all([
    fetchJson(window.SILICA_PREDICTIONS.chartsUrl),
    fetchJson(window.SILICA_PREDICTIONS.metricsUrl),
  ]);
  predictionState.charts = chartPayload.charts;
  predictionState.metrics = metricsPayload.metrics;
  const selectedChart = resolveSelectedPredictionChart();
  if (!selectedChart) {
    showMissingPredictionChart();
    return;
  }
  await renderPredictionChart(selectedChart.id);
}

function populatePredictionExperimentSelect(experiments) {
  const select = document.getElementById("predictionExperimentSelect");
  select.replaceChildren();
  for (const experiment of experiments) {
    const option = document.createElement("option");
    option.value = experiment;
    option.textContent = experiment;
    select.appendChild(option);
  }
  if (
    predictionState.selectedExperiment &&
    !experiments.includes(predictionState.selectedExperiment)
  ) {
    const option = document.createElement("option");
    option.value = predictionState.selectedExperiment;
    option.textContent = predictionState.selectedExperiment;
    option.disabled = true;
    select.appendChild(option);
  }
  select.value = predictionState.selectedExperiment;
}

function bindPredictionControls() {
  const select = document.getElementById("predictionExperimentSelect");
  select.addEventListener("pointerdown", revealPredictionTopbarSelectionLabels);
  select.addEventListener("focus", revealPredictionTopbarSelectionLabels);
  select.addEventListener("change", (event) => {
    revealPredictionTopbarSelectionLabels();
    predictionState.selectedExperiment = event.target.value;
    if (!predictionState.experiments.includes(predictionState.selectedExperiment)) {
      throw new Error(`Unknown experiment: ${predictionState.selectedExperiment}`);
    }
    syncPredictionExperimentQuery();
    const selectedChart = resolveSelectedPredictionChart();
    if (!selectedChart) {
      showMissingPredictionChart();
      return;
    }
    void renderPredictionChart(selectedChart.id);
  });
}

function showMissingPredictionChart() {
  cancelPendingPredictionChartRender();
  predictionState.chartLoadToken += 1;
  predictionState.selectedChartId = "";
  clearPredictionMetrics();
  clearPredictionChart();
  setPredictionTopbarStatus("error");
  setPredictionStatus(
    `No prediction chart is configured for experiment: ${predictionState.selectedExperiment}`,
  );
}

function resolveInitialPredictionExperiment(defaultExperiment) {
  const searchParams = new URLSearchParams(window.location.search);
  const requestedExperiment = searchParams.get(EXPERIMENT_QUERY_PARAM) ?? "";
  if (predictionState.experiments.includes(requestedExperiment)) {
    return requestedExperiment;
  }
  if (requestedExperiment) {
    return requestedExperiment;
  }
  if (predictionState.experiments.includes(defaultExperiment)) {
    return defaultExperiment;
  }
  throw new Error(`Unknown default experiment: ${defaultExperiment}`);
}

function revealPredictionTopbarSelectionLabels() {
  SilicaUI.revealTopbarSelectionLabels(predictionState);
}

function bindPredictionTopbarSelectionReveal() {
  SilicaUI.bindTopbarSelectionReveal({
    state: predictionState,
    triggerSelectors: [".topbar-experiment-pill"],
  });
}

async function renderPredictionChart(
  chartId,
  { renderMetrics = true, updateStatus = true } = {},
) {
  if (updateStatus) {
    cancelPendingPredictionChartRender();
  }
  const loadToken = predictionState.chartLoadToken + 1;
  predictionState.chartLoadToken = loadToken;
  predictionState.selectedChartId = chartId;
  if (updateStatus) {
    setPredictionTopbarStatus("loading");
    setPredictionStatus("Loading chart...");
  }
  try {
    const spec = await fetchJson(predictionChartSpecUrl(chartId));
    spec.width = "container";
    spec.height = "container";
    spec.autosize = {
      ...(spec.autosize ?? {}),
      type: "fit",
      contains: "padding",
      resize: true,
    };
    applyPredictionChartYAxisDomain(spec);
    applyPredictionChartClipping(spec);
    if (loadToken !== predictionState.chartLoadToken) {
      return;
    }
    if (predictionState.chartView) {
      predictionState.chartView.finalize();
      predictionState.chartView = null;
    }
    const result = await vegaEmbed("#predictionChart", spec, {
      actions: false,
      mode: "vega-lite",
    });
    if (loadToken !== predictionState.chartLoadToken) {
      result.view.finalize();
      return;
    }
    predictionState.chartView = result.view;
    result.view.addEventListener("click", (_event, item) => {
      handlePredictionChartClick(item);
    });
    if (updateStatus) {
      setPredictionStatus("");
    }
    refreshPredictionChartLayout();
    const metricsLoaded = renderMetrics
      ? await renderPredictionMetrics(loadToken)
      : true;
    if (loadToken !== predictionState.chartLoadToken) {
      return;
    }
    if (metricsLoaded && updateStatus) {
      setPredictionTopbarStatus("ready");
    }
  } catch (error) {
    console.error(error);
    if (loadToken !== predictionState.chartLoadToken) {
      return;
    }
    if (updateStatus) {
      setPredictionTopbarStatus("error");
    }
    setPredictionStatus(String(error));
    if (renderMetrics) {
      setPredictionMetricsStatus(String(error));
    }
  }
}

function applyPredictionChartYAxisDomain(spec) {
  const expandedDomain =
    predictionState.yDomain ??
    expandPredictionChartYDomain(collectPredictionChartYDomain(spec, spec.data, spec));
  applyPredictionChartYAxisDomainToSpec(spec, expandedDomain);
}

function applyPredictionChartClipping(spec) {
  if (spec.mark) {
    if (typeof spec.mark === "string") {
      spec.mark = { type: spec.mark, clip: true };
    } else {
      spec.mark = { ...spec.mark, clip: true };
    }
  }
  for (const childSpec of spec.layer ?? []) {
    applyPredictionChartClipping(childSpec);
  }
}

function applyPredictionChartYAxisDomainToSpec(spec, domain) {
  if (spec.encoding?.y) {
    spec.encoding.y.scale = {
      ...(spec.encoding.y.scale ?? {}),
      domain,
    };
  }
  for (const childSpec of spec.layer ?? []) {
    applyPredictionChartYAxisDomainToSpec(childSpec, domain);
  }
}

function expandPredictionChartYDomain(currentDomain) {
  if (!currentDomain) {
    return PREDICTION_CHART_Y_DOMAIN;
  }
  return [
    Math.min(PREDICTION_CHART_Y_DOMAIN[0], currentDomain[0]),
    Math.max(PREDICTION_CHART_Y_DOMAIN[1], currentDomain[1]),
  ];
}

function collectPredictionChartYDomain(spec, inheritedData = null, rootSpec = spec) {
  const domains = [];
  const ownDomain = predictionChartSpecYDomain(spec, inheritedData, rootSpec);
  if (ownDomain) {
    domains.push(ownDomain);
  }

  const childInheritedData = spec.data ?? inheritedData;
  for (const childSpec of spec.layer ?? []) {
    const childDomain = collectPredictionChartYDomain(
      childSpec,
      childInheritedData,
      rootSpec,
    );
    if (childDomain) {
      domains.push(childDomain);
    }
  }

  if (domains.length === 0) {
    return null;
  }
  return [
    Math.min(...domains.map((domain) => domain[0])),
    Math.max(...domains.map((domain) => domain[1])),
  ];
}

function predictionChartSpecYDomain(spec, inheritedData, rootSpec) {
  const scaleDomain = spec.encoding?.y?.scale?.domain;
  if (isNumericDomain(scaleDomain)) {
    return scaleDomain.map(Number);
  }

  const fieldName = spec.encoding?.y?.field;
  if (!fieldName) {
    return null;
  }

  const dataValues = predictionChartDataValues(spec.data ?? inheritedData, rootSpec);
  const numericValues = dataValues
    .map((datum) => Number(datum[fieldName]))
    .filter(Number.isFinite);
  if (numericValues.length === 0) {
    return null;
  }
  return [Math.min(...numericValues), Math.max(...numericValues)];
}

function predictionChartDataValues(data, rootSpec) {
  if (Array.isArray(data?.values)) {
    return data.values;
  }
  if (data?.name && Array.isArray(rootSpec.datasets?.[data.name])) {
    return rootSpec.datasets[data.name];
  }
  return [];
}

function isNumericDomain(domain) {
  return (
    Array.isArray(domain) &&
    domain.length === 2 &&
    domain.every((value) => Number.isFinite(Number(value)))
  );
}

function resolveInitialPredictionYDomain() {
  const searchParams = new URLSearchParams(window.location.search);
  const yMin = searchParams.get(Y_MIN_QUERY_PARAM);
  const yMax = searchParams.get(Y_MAX_QUERY_PARAM);
  if (yMin === null && yMax === null) {
    return null;
  }
  if (yMin === null || yMax === null) {
    throw new Error("Prediction y-axis URL bounds require both y_min and y_max");
  }
  const domain = [Number(yMin), Number(yMax)];
  if (!isNumericDomain(domain) || domain[0] >= domain[1]) {
    throw new Error(`Invalid prediction y-axis URL bounds: ${yMin}, ${yMax}`);
  }
  return domain;
}

function updatePredictionYDomain(domain) {
  if (!isNumericDomain(domain)) {
    return;
  }
  const nextDomain = domain.map(Number);
  if (nextDomain[0] >= nextDomain[1]) {
    return;
  }
  if (
    predictionState.yDomain &&
    predictionState.yDomain[0] === nextDomain[0] &&
    predictionState.yDomain[1] === nextDomain[1]
  ) {
    return;
  }
  predictionState.yDomain = nextDomain;
  syncPredictionExperimentQuery();
}

function bindPredictionChartYAxisInteractions() {
  const chartElement = document.getElementById("predictionChart");
  chartElement.addEventListener(
    "wheel",
    (event) => {
      if (!canScalePredictionChart()) {
        return;
      }
      event.preventDefault();
      zoomPredictionYAxisFromWheel(event);
    },
    { passive: false },
  );
  chartElement.addEventListener("pointerdown", (event) => {
    if (!canScalePredictionChart() || event.button !== 0) {
      return;
    }
    predictionState.yDragStart = {
      active: false,
      clientX: event.clientX,
      clientY: event.clientY,
      domain: currentPredictionYDomain(),
      pointerId: event.pointerId,
    };
  });
  chartElement.addEventListener("pointermove", (event) => {
    if (!predictionState.yDragStart) {
      return;
    }
    const dragDistance = Math.hypot(
      event.clientX - predictionState.yDragStart.clientX,
      event.clientY - predictionState.yDragStart.clientY,
    );
    if (!predictionState.yDragStart.active) {
      if (dragDistance < PREDICTION_Y_DRAG_THRESHOLD_PX) {
        return;
      }
      predictionState.yDragStart.active = true;
      chartElement.setPointerCapture(predictionState.yDragStart.pointerId);
    }
    event.preventDefault();
    panPredictionYAxisFromPointer(event);
  });
  chartElement.addEventListener("pointerup", clearPredictionYAxisDrag);
  chartElement.addEventListener("pointercancel", clearPredictionYAxisDrag);
  chartElement.addEventListener("lostpointercapture", clearPredictionYAxisDrag);
}

function zoomPredictionYAxisFromWheel(event) {
  const domain = currentPredictionYDomain();
  const chartElement = document.getElementById("predictionChart");
  const chartRect = chartElement.getBoundingClientRect();
  if (chartRect.height <= 0) {
    return;
  }

  const pointerRatio = Math.min(
    1,
    Math.max(0, (event.clientY - chartRect.top) / chartRect.height),
  );
  const anchor = domain[1] - (domain[1] - domain[0]) * pointerRatio;
  const zoomFactor = Math.pow(PREDICTION_Y_ZOOM_WHEEL_FACTOR, event.deltaY);
  const nextDomain = [
    anchor - (anchor - domain[0]) * zoomFactor,
    anchor + (domain[1] - anchor) * zoomFactor,
  ];
  updatePredictionYDomain(nextDomain);
  schedulePredictionChartRender();
}

function panPredictionYAxisFromPointer(event) {
  const chartElement = document.getElementById("predictionChart");
  const chartRect = chartElement.getBoundingClientRect();
  if (chartRect.height <= 0) {
    return;
  }

  const startDomain = predictionState.yDragStart.domain;
  const domainSpan = startDomain[1] - startDomain[0];
  const yDelta =
    ((event.clientY - predictionState.yDragStart.clientY) / chartRect.height) *
    domainSpan;
  updatePredictionYDomain([startDomain[0] + yDelta, startDomain[1] + yDelta]);
  schedulePredictionChartRender();
}

function clearPredictionYAxisDrag() {
  if (predictionState.yDragStart?.active) {
    predictionState.suppressPredictionClickUntil =
      performance.now() + PREDICTION_DRAG_CLICK_SUPPRESSION_MS;
  }
  predictionState.yDragStart = null;
}

function currentPredictionYDomain() {
  return predictionState.yDomain ?? PREDICTION_CHART_Y_DOMAIN;
}

function canScalePredictionChart() {
  return (
    Boolean(predictionState.chartView) && predictionState.topbarStatus !== "loading"
  );
}

function schedulePredictionChartRender() {
  if (!predictionState.selectedChartId || predictionState.topbarStatus === "loading") {
    return;
  }
  if (predictionState.pendingChartRenderFrame !== null) {
    window.cancelAnimationFrame(predictionState.pendingChartRenderFrame);
  }
  predictionState.pendingChartRenderFrame = window.requestAnimationFrame(() => {
    predictionState.pendingChartRenderFrame = null;
    void renderPredictionChart(predictionState.selectedChartId, {
      renderMetrics: false,
      updateStatus: false,
    });
  });
}

function cancelPendingPredictionChartRender() {
  if (predictionState.pendingChartRenderFrame === null) {
    return;
  }
  window.cancelAnimationFrame(predictionState.pendingChartRenderFrame);
  predictionState.pendingChartRenderFrame = null;
}

function syncPredictionExperimentQuery() {
  const url = new URL(window.location.href);
  url.searchParams.set(EXPERIMENT_QUERY_PARAM, predictionState.selectedExperiment);
  if (predictionState.yDomain) {
    url.searchParams.set(
      Y_MIN_QUERY_PARAM,
      formatPredictionYDomainParam(predictionState.yDomain[0]),
    );
    url.searchParams.set(
      Y_MAX_QUERY_PARAM,
      formatPredictionYDomainParam(predictionState.yDomain[1]),
    );
  } else {
    url.searchParams.delete(Y_MIN_QUERY_PARAM);
    url.searchParams.delete(Y_MAX_QUERY_PARAM);
  }
  window.history.replaceState({}, "", url);
}

function formatPredictionYDomainParam(value) {
  return Number(value).toPrecision(8);
}

function handlePredictionChartClick(item) {
  if (performance.now() < predictionState.suppressPredictionClickUntil) {
    return;
  }
  if (!item) {
    return;
  }
  const datum = resolvePredictionClickDatum(item);
  if (!datum) {
    return;
  }
  const slideKey = resolvePredictionSlideKey(datum);
  if (!slideKey) {
    return;
  }
  const url = new URL(window.SILICA_PREDICTIONS.slideViewerUrl, window.location.origin);
  url.searchParams.set(SLIDE_QUERY_PARAM, slideKey);
  url.searchParams.set(EXPERIMENT_QUERY_PARAM, predictionState.selectedExperiment);
  window.location.href = url.toString();
}

function resolvePredictionClickDatum(item) {
  const seen = new Set();
  const stack = [item];
  while (stack.length > 0) {
    const datum = stack.pop();
    if (!datum || typeof datum !== "object" || seen.has(datum)) {
      continue;
    }
    seen.add(datum);
    if (isPredictionChartDatum(datum)) {
      return datum;
    }
    for (const value of Object.values(datum)) {
      if (value && typeof value === "object") {
        stack.push(value);
      }
    }
  }
  return null;
}

function isPredictionChartDatum(datum) {
  return (
    datum.raw_score !== undefined &&
    datum.label !== undefined &&
    (datum.slide_id !== undefined || datum.path !== undefined)
  );
}

function resolvePredictionSlideKey(datum) {
  if (datum.slide_id !== null && datum.slide_id !== undefined) {
    const slideKey = String(datum.slide_id).trim();
    if (slideKey) {
      return slideKey;
    }
  }
  if (datum.path !== null && datum.path !== undefined) {
    const slideKey = String(datum.path)
      .trim()
      .replace(/[\\/]+/g, "-")
      .replace(/\s+/g, "");
    if (slideKey) {
      return slideKey;
    }
  }
  return null;
}

function predictionChartSpecUrl(chartId) {
  return `${window.SILICA_PREDICTIONS.chartsUrl}/${chartId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/")}`;
}

async function renderPredictionMetrics(loadToken = predictionState.chartLoadToken) {
  if (loadToken !== predictionState.chartLoadToken) {
    return false;
  }
  clearPredictionMetrics();
  const metrics = resolveSelectedMetrics();
  if (!metrics) {
    if (loadToken !== predictionState.chartLoadToken) {
      return false;
    }
    setPredictionTopbarStatus("error");
    setPredictionMetricsStatus(
      `No prediction metrics are configured for experiment: ${predictionState.selectedExperiment}`,
    );
    return false;
  }

  predictionState.selectedMetricsId = metrics.id;
  setPredictionMetricsStatus("Loading metrics...");
  const payload = await fetchJson(predictionMetricsPayloadUrl(metrics.id));
  if (loadToken !== predictionState.chartLoadToken) {
    return false;
  }
  setMetricText("metricNumDatapoints", formatCount(payload.num_datapoints));
  setMetricText(
    "metricPredLabelMeanClassAccuracy",
    formatPercent(payload.pred_label_mean_class_accuracy),
  );
  setMetricText("metricRawScorePearson", formatNumber(payload.raw_score_pearson));
  setMetricText("metricRawScoreSpearman", formatNumber(payload.raw_score_spearman));
  setMetricText("metricAuroc0Vs123", formatNumber(payload.auroc_0_vs_123));
  setMetricText("metricAuroc01Vs23", formatNumber(payload.auroc_01_vs_23));
  setMetricText("metricAuroc012Vs3", formatNumber(payload.auroc_012_vs_3));
  renderConfusionMatrix("predictionConfusionMatrix", payload.confusion_matrix);
  setPredictionMetricsStatus("");
  return true;
}

function resolveSelectedMetrics() {
  return predictionState.metrics.find(
    (metrics) => metrics.experiment === predictionState.selectedExperiment,
  );
}

function resolveSelectedPredictionChart() {
  return predictionState.charts.find(
    (chart) => chart.experiment === predictionState.selectedExperiment,
  );
}

function renderConfusionMatrix(containerId, matrix) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!Array.isArray(matrix) || matrix.length === 0) {
    container.textContent = "-";
    return;
  }

  const numCols = matrix[0].length;
  const numRows = matrix.length;
  const classLabels = Array.from({ length: numCols }, (_, i) => String(i));

  const wrap = document.createElement("div");
  wrap.className = "confusion-matrix-wrap";

  const yAxisTitle = document.createElement("div");
  yAxisTitle.className = "confusion-axis-title confusion-axis-title-y";
  yAxisTitle.textContent = "True";
  wrap.appendChild(yAxisTitle);

  const yTicks = document.createElement("div");
  yTicks.className = "confusion-ticks confusion-ticks-y";
  yTicks.style.gridTemplateRows = `repeat(${numRows}, minmax(0, 1fr))`;
  for (let r = 0; r < numRows; r++) {
    const tick = document.createElement("div");
    tick.className = "confusion-tick";
    tick.textContent = classLabels[r] ?? String(r);
    yTicks.appendChild(tick);
  }
  wrap.appendChild(yTicks);

  const xTicks = document.createElement("div");
  xTicks.className = "confusion-ticks confusion-ticks-x";
  xTicks.style.gridTemplateColumns = `repeat(${numCols}, minmax(0, 1fr))`;
  for (let c = 0; c < numCols; c++) {
    const tick = document.createElement("div");
    tick.className = "confusion-tick";
    tick.textContent = classLabels[c];
    xTicks.appendChild(tick);
  }
  wrap.appendChild(xTicks);

  const grid = document.createElement("div");
  grid.className = "confusion-matrix-grid";
  grid.style.gridTemplateColumns = `repeat(${numCols}, minmax(0, 1fr))`;
  grid.style.gridTemplateRows = `repeat(${numRows}, minmax(0, 1fr))`;
  for (const row of matrix) {
    for (const value of row) {
      const cell = document.createElement("div");
      cell.className = "confusion-cell";
      cell.textContent = formatCount(value);
      grid.appendChild(cell);
    }
  }
  wrap.appendChild(grid);

  const xAxisTitle = document.createElement("div");
  xAxisTitle.className = "confusion-axis-title confusion-axis-title-x";
  xAxisTitle.textContent = "Predicted";
  wrap.appendChild(xAxisTitle);

  container.appendChild(wrap);
}

function clearPredictionMetrics() {
  for (const id of [
    "metricNumDatapoints",
    "metricPredLabelMeanClassAccuracy",
    "metricRawScorePearson",
    "metricRawScoreSpearman",
    "metricAuroc0Vs123",
    "metricAuroc01Vs23",
    "metricAuroc012Vs3",
  ]) {
    setMetricText(id, "-");
  }
  renderConfusionMatrix("predictionConfusionMatrix", []);
}

function clearPredictionChart() {
  if (predictionState.chartView) {
    predictionState.chartView.finalize();
  }
  predictionState.chartView = null;
  document.getElementById("predictionChart").replaceChildren();
}

function predictionMetricsPayloadUrl(metricsId) {
  return `${window.SILICA_PREDICTIONS.metricsUrl}/${metricsId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/")}`;
}

function setMetricText(id, value) {
  document.getElementById(id).textContent = value;
}

function formatCount(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(
    numberValue,
  );
}

function formatPercent(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) {
    return "-";
  }
  return `${(numberValue * 100).toFixed(1)}%`;
}

function formatNumber(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) {
    return "-";
  }
  return numberValue.toFixed(3);
}

function bindPredictionSidebarControls() {
  SilicaUI.bindSidebarCollapse({
    state: predictionState,
    buttonIds: ["predictionSidebarCollapseButton"],
    onChange: refreshPredictionChartLayout,
  });

  SilicaUI.bindLayoutResize({
    state: predictionState,
    handleId: "predictionLayoutResizeHandle",
    updateFromPointer: updatePredictionSidebarWidthFromPointer,
    onEnd: refreshPredictionChartLayout,
  });

  bindPredictionResponsiveControls();
  window.addEventListener("resize", refreshPredictionChartLayout);
}

function updatePredictionSidebarWidthFromPointer(clientX) {
  const layout = document.querySelector(".predictions-layout");
  const layoutRect = layout.getBoundingClientRect();
  const sidebarWidthPercent =
    ((layoutRect.right - clientX) / Math.max(layoutRect.width, 1)) * 100;
  const clampedWidth = Math.min(42, Math.max(18, sidebarWidthPercent));
  document.documentElement.style.setProperty(
    "--sidebar-width",
    `${clampedWidth.toFixed(2)}%`,
  );
  refreshPredictionChartLayout();
}

function bindPredictionResponsiveControls() {
  SilicaUI.bindResponsivePlacement({
    element: document.querySelector(".topbar-experiment-picker"),
    mobileSlot: document.getElementById("predictionMobileExperimentPickerSlot"),
    desktopParent: document.querySelector(".topbar-inner"),
    mediaQueryText: "(max-width: 1024px)",
    anchorLabel: "prediction experiment picker desktop anchor",
    onChange: refreshPredictionChartLayout,
  });
}

function refreshPredictionChartLayout() {
  if (!predictionState.chartView) {
    return;
  }
  window.requestAnimationFrame(() => {
    const chartElement = document.getElementById("predictionChart");
    const chartRect = chartElement.getBoundingClientRect();
    if (chartRect.width <= 0 || chartRect.height <= 0) {
      return;
    }
    predictionState.chartView
      .width(Math.floor(chartRect.width))
      .height(Math.floor(chartRect.height))
      .resize();
    void predictionState.chartView.runAsync();
  });
}

function setPredictionTopbarStatus(status) {
  SilicaUI.setStatusPill({
    state: predictionState,
    status,
    pillSelector: ".topbar-experiment-pill",
    statusName: "prediction topbar",
  });
  if (
    status === "error" &&
    !document.getElementById("predictionDiagnosticsStatus").textContent
  ) {
    setPredictionDiagnosticsStatus("The predictions page encountered an error.");
  }
}

function setPredictionStatus(message) {
  setPredictionDiagnosticsStatus(message);
}

function setPredictionMetricsStatus(message) {
  setPredictionDiagnosticsStatus(message);
}

function setPredictionDiagnosticsStatus(message) {
  document.getElementById("predictionDiagnosticsStatus").textContent = message;
}
