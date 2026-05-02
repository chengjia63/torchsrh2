const predictionState = {
  charts: [],
  metrics: [],
  experiments: [],
  selectedChartId: "",
  selectedExperiment: "",
  selectedMetricsId: "",
  chartView: null,
  invalidRequestedExperiment: "",
  topbarStatus: "idle",
  topbarStatusHideTimer: null,
  sidebarCollapsed: false,
  layoutResizeActive: false,
  topbarSelectionRevealed: false,
};
const PREDICTION_CHART_Y_DOMAIN = [-5, 6];

document.addEventListener("DOMContentLoaded", () => {
  void bootstrapPredictions().catch((error) => {
    console.error(error);
    setPredictionTopbarStatus("error");
    setPredictionStatus(String(error));
  });
});

async function bootstrapPredictions() {
  const slidePayload = await fetchJson(window.SILICA_PREDICTIONS.slidesUrl);
  predictionState.experiments = slidePayload.experiments ?? [];
  predictionState.selectedExperiment = resolveInitialPredictionExperiment(
    slidePayload.default_experiment ?? "",
  );
  populatePredictionExperimentSelect(predictionState.experiments);
  bindPredictionControls();
  bindPredictionTopbarSelectionReveal();
  bindPredictionSidebarControls();
  if (predictionState.invalidRequestedExperiment) {
    syncPredictionExperimentQuery();
    clearPredictionMetrics();
    clearPredictionChart();
    setPredictionTopbarStatus("error");
    setPredictionStatus(
      `Unknown experiment: ${predictionState.invalidRequestedExperiment}`,
    );
    return;
  }
  syncPredictionExperimentQuery();

  setPredictionTopbarStatus("loading");
  setPredictionStatus("Loading charts...");
  const [chartPayload, metricsPayload] = await Promise.all([
    fetchJson(window.SILICA_PREDICTIONS.chartsUrl),
    fetchJson(window.SILICA_PREDICTIONS.metricsUrl),
  ]);
  predictionState.charts = chartPayload.charts ?? [];
  predictionState.metrics = metricsPayload.metrics ?? [];
  const selectedChart = resolveSelectedPredictionChart();
  if (!selectedChart) {
    clearPredictionMetrics();
    clearPredictionChart();
    setPredictionTopbarStatus("error");
    setPredictionStatus(
      `No prediction chart is configured for experiment: ${predictionState.selectedExperiment}`,
    );
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
    predictionState.invalidRequestedExperiment &&
    !experiments.includes(predictionState.invalidRequestedExperiment)
  ) {
    const option = document.createElement("option");
    option.value = predictionState.invalidRequestedExperiment;
    option.textContent = predictionState.invalidRequestedExperiment;
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
      predictionState.invalidRequestedExperiment = predictionState.selectedExperiment;
      syncPredictionExperimentQuery();
      clearPredictionMetrics();
      clearPredictionChart();
      setPredictionTopbarStatus("error");
      setPredictionStatus(`Unknown experiment: ${predictionState.selectedExperiment}`);
      return;
    }
    predictionState.invalidRequestedExperiment = "";
    syncPredictionExperimentQuery();
    const selectedChart = resolveSelectedPredictionChart();
    if (!selectedChart) {
      clearPredictionMetrics();
      clearPredictionChart();
      setPredictionTopbarStatus("error");
      setPredictionStatus(
        `No prediction chart is configured for experiment: ${predictionState.selectedExperiment}`,
      );
      return;
    }
    void renderPredictionChart(selectedChart.id);
  });
}

function resolveInitialPredictionExperiment(defaultExperiment) {
  const searchParams = new URLSearchParams(window.location.search);
  const requestedExperiment = searchParams.get("experiment") ?? "";
  if (predictionState.experiments.includes(requestedExperiment)) {
    return requestedExperiment;
  }
  if (requestedExperiment) {
    predictionState.invalidRequestedExperiment = requestedExperiment;
    return requestedExperiment;
  }
  if (predictionState.experiments.includes(defaultExperiment)) {
    return defaultExperiment;
  }
  if (defaultExperiment) {
    predictionState.invalidRequestedExperiment = defaultExperiment;
    return defaultExperiment;
  }
  predictionState.invalidRequestedExperiment = "default experiment";
  return "";
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
      const datum = resolveClickedDatum(item);
      if (!datum) {
        return;
      }
      const slideKey = datum.slide ?? datum.Slide ?? datum.path;
      if (!slideKey) {
        return;
      }
      openSlideViewer(String(slideKey));
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
  const experiment =
    predictionState.invalidRequestedExperiment || predictionState.selectedExperiment;
  if (experiment) {
    url.searchParams.set("experiment", experiment);
  } else {
    url.searchParams.delete("experiment");
  }
  window.history.replaceState({}, "", url);
}

function resolveClickedDatum(item) {
  let cursor = item;
  while (cursor) {
    if (cursor.datum) {
      return cursor.datum.datum ?? cursor.datum;
    }
    cursor = cursor.mark?.group;
  }
  return null;
}

function openSlideViewer(slideKey) {
  const url = new URL(window.SILICA_PREDICTIONS.slideViewerUrl, window.location.origin);
  url.searchParams.set("slide", slideKey);
  if (predictionState.selectedExperiment) {
    url.searchParams.set("experiment", predictionState.selectedExperiment);
  }
  window.location.href = url.toString();
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
  setMetricText(
    "metricSigmoidSumPredLabelMeanClassAccuracy",
    formatPercent(payload.sigmoid_sum_pred_label_mean_class_accuracy),
  );
  setMetricText("metricRawScorePearson", formatNumber(payload.raw_score_pearson));
  setMetricText(
    "metricSigmoidSumScorePearson",
    formatNumber(payload.sigmoid_sum_score_pearson),
  );
  setMetricText("metricRawScoreSpearman", formatNumber(payload.raw_score_spearman));
  setMetricText(
    "metricSigmoidSumScoreSpearman",
    formatNumber(payload.sigmoid_sum_score_spearman),
  );
  setMetricText("metricAuroc0Vs123", formatNumber(payload.auroc_0_vs_123));
  setMetricText("metricAuroc01Vs23", formatNumber(payload.auroc_01_vs_23));
  setMetricText("metricAuroc012Vs3", formatNumber(payload.auroc_012_vs_3));
  renderConfusionMatrix("predictionConfusionMatrix", payload.confusion_matrix);
  renderConfusionMatrix(
    "sigmoidSumConfusionMatrix",
    payload.sigmoid_sum_confusion_matrix,
  );
  setPredictionMetricsStatus("");
  return true;
}

function resolveSelectedMetrics() {
  if (predictionState.metrics.length === 0) {
    return null;
  }

  return predictionState.metrics.find((metrics) =>
    predictionRecordMatchesExperiment(metrics, predictionState.selectedExperiment),
  ) ?? null;
}

function resolveSelectedPredictionChart() {
  return predictionState.charts.find((chart) =>
    predictionRecordMatchesExperiment(chart, predictionState.selectedExperiment),
  ) ?? null;
}

function predictionRecordMatchesExperiment(record, experiment) {
  return (
    record.id === experiment ||
    record.filename === experiment ||
    record.label === experiment ||
    record.experiment === experiment
  );
}

function renderConfusionMatrix(containerId, matrix) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!Array.isArray(matrix) || matrix.length === 0) {
    container.textContent = "-";
    return;
  }

  const table = document.createElement("table");
  const tbody = document.createElement("tbody");
  for (const row of matrix) {
    const tr = document.createElement("tr");
    for (const value of row) {
      const td = document.createElement("td");
      td.textContent = formatCount(value);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}

function clearPredictionMetrics() {
  for (const id of [
    "metricNumDatapoints",
    "metricPredLabelMeanClassAccuracy",
    "metricSigmoidSumPredLabelMeanClassAccuracy",
    "metricRawScorePearson",
    "metricSigmoidSumScorePearson",
    "metricRawScoreSpearman",
    "metricSigmoidSumScoreSpearman",
    "metricAuroc0Vs123",
    "metricAuroc01Vs23",
    "metricAuroc012Vs3",
  ]) {
    setMetricText(id, "-");
  }
  renderConfusionMatrix("predictionConfusionMatrix", []);
  renderConfusionMatrix("sigmoidSumConfusionMatrix", []);
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
