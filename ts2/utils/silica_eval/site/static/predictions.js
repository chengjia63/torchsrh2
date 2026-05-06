const predictionState = {
  charts: [],
  metrics: [],
  experiments: [],
  selectedChartId: "",
  selectedExperiment: "",
  selectedMetricsId: "",
  chartView: null,
  topbarStatus: "idle",
  topbarStatusHideTimer: null,
  sidebarCollapsed: false,
  layoutResizeActive: false,
  topbarSelectionRevealed: false,
};
const PREDICTION_CHART_Y_DOMAIN = [-5, 6];
const EXPERIMENT_QUERY_PARAM = "experiment";
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
  populatePredictionExperimentSelect(predictionState.experiments);
  bindPredictionControls();
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
    throw new Error(`Unknown experiment: ${requestedExperiment}`);
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

async function renderPredictionChart(chartId) {
  predictionState.selectedChartId = chartId;
  setPredictionTopbarStatus("loading");
  setPredictionStatus("Loading chart...");
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
    const result = await vegaEmbed("#predictionChart", spec, {
      actions: false,
      mode: "vega-lite",
    });
    predictionState.chartView = result.view;
    result.view.addEventListener("click", (_event, item) => {
      handlePredictionChartClick(item);
    });
    setPredictionStatus("");
    refreshPredictionChartLayout();
    const metricsLoaded = await renderPredictionMetrics();
    if (metricsLoaded) {
      setPredictionTopbarStatus("ready");
    }
  } catch (error) {
    console.error(error);
    setPredictionTopbarStatus("error");
    setPredictionStatus(String(error));
    setPredictionMetricsStatus(String(error));
  }
}

function applyPredictionChartYAxisDomain(spec) {
  const expandedDomain = expandPredictionChartYDomain(
    collectPredictionChartYDomain(spec, spec.data, spec),
  );
  applyPredictionChartYAxisDomainToSpec(spec, expandedDomain);
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

function syncPredictionExperimentQuery() {
  const url = new URL(window.location.href);
  url.searchParams.set(EXPERIMENT_QUERY_PARAM, predictionState.selectedExperiment);
  window.history.replaceState({}, "", url);
}

function handlePredictionChartClick(item) {
  if (!item || !item.datum) {
    return;
  }
  const datum = resolvePredictionClickDatum(item);
  const slideKey = resolvePredictionSlideKey(datum);
  const url = new URL(window.SILICA_PREDICTIONS.slideViewerUrl, window.location.origin);
  url.searchParams.set(SLIDE_QUERY_PARAM, slideKey);
  url.searchParams.set(EXPERIMENT_QUERY_PARAM, predictionState.selectedExperiment);
  window.location.href = url.toString();
}

function resolvePredictionClickDatum(item) {
  const datum = item.datum;
  if (datum.slide_id !== undefined || datum.path !== undefined) {
    return datum;
  }
  if (
    datum.datum &&
    (datum.datum.slide_id !== undefined || datum.datum.path !== undefined)
  ) {
    return datum.datum;
  }
  const message = `Unable to resolve prediction datum from Vega item fields: ${Object.keys(datum).join(", ")}`;
  setPredictionTopbarStatus("error");
  setPredictionStatus(message);
  throw new Error(message);
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
  const message = `Unable to resolve slide key from prediction datum fields: ${Object.keys(datum).join(", ")}`;
  setPredictionTopbarStatus("error");
  setPredictionStatus(message);
  throw new Error(message);
}

function predictionChartSpecUrl(chartId) {
  return `${window.SILICA_PREDICTIONS.chartsUrl}/${chartId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/")}`;
}

async function renderPredictionMetrics() {
  clearPredictionMetrics();
  const metrics = resolveSelectedMetrics();
  if (!metrics) {
    setPredictionTopbarStatus("error");
    setPredictionMetricsStatus(
      `No prediction metrics are configured for experiment: ${predictionState.selectedExperiment}`,
    );
    return false;
  }

  predictionState.selectedMetricsId = metrics.id;
  setPredictionMetricsStatus("Loading metrics...");
  const payload = await fetchJson(predictionMetricsPayloadUrl(metrics.id));
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
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

function setPredictionStatus(message) {
  document.getElementById("predictionChartStatus").textContent = message;
}

function setPredictionMetricsStatus(message) {
  document.getElementById("predictionMetricsStatus").textContent = message;
}
