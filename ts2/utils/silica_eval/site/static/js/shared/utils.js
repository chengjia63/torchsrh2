const SilicaSiteUtils = (() => {
  const {
    CHART_AXIS_STROKE_STYLE,
    CHART_BACKGROUND_STYLE,
    CHART_GRID_STROKE_STYLE,
    CHART_TEXT_FILL_STYLE,
  } = SilicaSiteConfig;

  function clampNumber(value, minValue, maxValue) {
    return Math.min(Math.max(value, minValue), maxValue);
  }

  function formatInteger(value) {
    return new Intl.NumberFormat("en-US").format(value);
  }

  async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status}`);
    }
    return response.json();
  }

  function colorFromStops(value, stops) {
    const clamped = clampNumber(value, 0, 1);
    let leftStop = stops[0];
    let rightStop = stops[stops.length - 1];
    for (let index = 1; index < stops.length; index += 1) {
      if (clamped <= stops[index].position) {
        leftStop = stops[index - 1];
        rightStop = stops[index];
        break;
      }
    }
    const segmentWidth = Math.max(1.0e-6, rightStop.position - leftStop.position);
    const t = (clamped - leftStop.position) / segmentWidth;
    const rgb = leftStop.color.map((channel, index) =>
      Math.round(channel + (rightStop.color[index] - channel) * t),
    );
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const devicePixelRatio = window.devicePixelRatio || 1;
    const nextWidth = Math.max(1, Math.round(rect.width * devicePixelRatio));
    const nextHeight = Math.max(1, Math.round(rect.height * devicePixelRatio));
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    return { width: rect.width, height: rect.height };
  }

  function buildHistogram(values, binCount, minValue, maxValue) {
    const bins = Array.from({ length: binCount }, (_, index) => ({
      start: minValue + ((maxValue - minValue) * index) / binCount,
      end: minValue + ((maxValue - minValue) * (index + 1)) / binCount,
      count: 0,
    }));
    if (!values.length) {
      return bins;
    }

    const span = maxValue - minValue;
    for (const rawValue of values) {
      const value = clampNumber(rawValue, minValue, maxValue - Number.EPSILON);
      const binIndex = Math.min(
        binCount - 1,
        Math.floor(((value - minValue) / span) * binCount),
      );
      bins[binIndex].count += 1;
    }
    return bins;
  }

  function buildClusterXAxisTickIndices(clusterCount) {
    if (clusterCount <= 18) {
      return Array.from({ length: clusterCount }, (_, index) => index);
    }
    const lastIndex = clusterCount - 1;
    const tickCount = 5;
    return Array.from(
      new Set(
        Array.from({ length: tickCount }, (_, index) =>
          Math.round((lastIndex * index) / (tickCount - 1)),
        ),
      ),
    );
  }

  function drawBarChart({
    context,
    width,
    height,
    xTickLabels,
    xTickIndices = null,
    values,
    fillStyle,
    emptyLabel,
  }) {
    context.save();
    context.fillStyle = CHART_BACKGROUND_STYLE;
    context.fillRect(0, 0, width, height);

    context.font = "11px 'Roboto Mono', monospace";
    const maxValue = Math.max(...values, 0);
    const yAxisLabelWidth = Math.ceil(
      context.measureText(formatChartTick(maxValue)).width,
    );
    const padding = {
      top: 10,
      right: 8,
      bottom: 34,
      left: Math.max(28, yAxisLabelWidth + 12),
    };
    const chartWidth = Math.max(0, width - padding.left - padding.right);
    const chartHeight = Math.max(0, height - padding.top - padding.bottom);
    if (chartWidth === 0 || chartHeight === 0) {
      context.restore();
      return;
    }

    const yTickCount = maxValue <= 4 ? Math.max(2, maxValue) : 4;

    context.strokeStyle = CHART_GRID_STROKE_STYLE;
    context.lineWidth = 1;
    for (let tickIndex = 0; tickIndex <= yTickCount; tickIndex += 1) {
      const tickValue = (maxValue * tickIndex) / yTickCount;
      const y = padding.top + chartHeight - (chartHeight * tickIndex) / yTickCount;
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(padding.left + chartWidth, y);
      context.stroke();

      context.fillStyle = CHART_TEXT_FILL_STYLE;
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(formatChartTick(tickValue), padding.left - 6, y);
    }

    context.strokeStyle = CHART_AXIS_STROKE_STYLE;
    context.lineWidth = 1.2;
    context.beginPath();
    context.moveTo(padding.left, padding.top);
    context.lineTo(padding.left, padding.top + chartHeight);
    context.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    context.stroke();

    if (maxValue === 0) {
      context.fillStyle = CHART_TEXT_FILL_STYLE;
      context.font = "500 13px Roboto, sans-serif";
      context.textAlign = "left";
      context.textBaseline = "middle";
      context.fillText(emptyLabel, padding.left + 8, padding.top + chartHeight / 2);
      context.restore();
      return;
    }

    const barCount = values.length;
    const sideInset = barCount > 24 ? 1 : 3;
    const gap = barCount > 80 ? 0 : barCount > 30 ? 1 : barCount > 14 ? 2 : 3;
    const plotWidth = Math.max(0, chartWidth - sideInset * 2);
    const rawBarWidth = (plotWidth - gap * Math.max(0, barCount - 1)) / barCount;
    const barWidth = barCount > 80 ? rawBarWidth : Math.max(1, rawBarWidth);

    for (let index = 0; index < values.length; index += 1) {
      const value = values[index];
      const barHeight = (value / maxValue) * chartHeight;
      const x = padding.left + sideInset + index * (barWidth + gap);
      const y = padding.top + chartHeight - barHeight;
      context.fillStyle = fillStyle;
      context.fillRect(x, y, barWidth, barHeight);
    }

    context.fillStyle = CHART_TEXT_FILL_STYLE;
    context.font = "11px 'Roboto Mono', monospace";
    context.textAlign = "center";
    context.textBaseline = "top";
    const defaultTickStep = barCount > 18 ? Math.ceil(barCount / 10) : 1;
    const tickIndices =
      xTickIndices ??
      Array.from(
        { length: Math.ceil(xTickLabels.length / defaultTickStep) },
        (_, index) => index * defaultTickStep,
      );
    for (const index of tickIndices) {
      const label = xTickLabels[index];
      const x =
        padding.left + sideInset + index * (barWidth + gap) + barWidth / 2;
      const axisY = padding.top + chartHeight;
      context.fillText(label, x, axisY + 8);
    }
    context.restore();
  }

  function formatChartTick(value) {
    if (value >= 100) {
      return `${Math.round(value)}`;
    }
    if (value >= 10) {
      return `${Math.round(value)}`;
    }
    if (value === 0) {
      return "0";
    }
    return `${Math.round(value * 10) / 10}`;
  }

  return Object.freeze({
    buildClusterXAxisTickIndices,
    buildHistogram,
    clampNumber,
    colorFromStops,
    drawBarChart,
    fetchJson,
    formatInteger,
    resizeCanvas,
  });
})();
