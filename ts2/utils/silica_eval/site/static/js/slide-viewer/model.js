const SilicaSlideModel = (() => {
  const { INFILTRATION_LABELS } = SilicaSiteConfig;

  function normalizeSlidesPayload(rawSlides) {
    if (!Array.isArray(rawSlides)) {
      throw new Error("/api/slides must return a slides array");
    }

    return rawSlides.map((slide) => ({
      ...slide,
      available_experiments: normalizeExperimentNames(slide.available_experiments),
      diagnosis: normalizeFilterValue(slide.diagnosis),
      infiltration: normalizeFilterValue(slide.infiltration),
    }));
  }

  function normalizeExperimentNames(values) {
    if (!Array.isArray(values)) {
      return [];
    }
    return sortFilterValues(
      values
        .map((value) => String(value ?? "").trim())
        .filter(Boolean),
    );
  }

  function resolveFilterOptions(rawFilters, slides) {
    const derivedDiagnosis = collectUniqueValues(slides, "diagnosis");
    const derivedInfiltration = collectUniqueValues(slides, "infiltration");

    return {
      diagnosis: Array.isArray(rawFilters?.diagnosis)
        ? sortFilterValues(rawFilters.diagnosis.map(normalizeFilterValue))
        : derivedDiagnosis,
      infiltration: Array.isArray(rawFilters?.infiltration)
        ? sortFilterValues(rawFilters.infiltration.map(normalizeFilterValue))
        : derivedInfiltration,
    };
  }

  function collectUniqueValues(slides, key) {
    return sortFilterValues(
      Array.from(new Set(slides.map((slide) => normalizeFilterValue(slide[key])))),
    );
  }

  function sortFilterValues(values) {
    return [...new Set(values)].sort((left, right) =>
      left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" }),
    );
  }

  function normalizeFilterValue(value) {
    if (value === null || value === undefined) {
      return "UNK";
    }
    const normalized = String(value).trim();
    return normalized || "UNK";
  }

  function getFilterDisplayLabel(filterKey, value, defaultLabel) {
    if (filterKey === "infiltration") {
      return INFILTRATION_LABELS[value] || value || defaultLabel;
    }
    return value || defaultLabel;
  }

  return Object.freeze({
    getFilterDisplayLabel,
    normalizeExperimentNames,
    normalizeFilterValue,
    normalizeSlidesPayload,
    resolveFilterOptions,
    sortFilterValues,
  });
})();
