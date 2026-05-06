const SilicaSlideSelection = (() => {
  function getSlideEntry(slides, slideKey) {
    return slides.find((slide) => slide.key === slideKey) || null;
  }

  function getSlideAvailableExperiments(slides, slideKey) {
    return getSlideEntry(slides, slideKey)?.available_experiments ?? [];
  }

  function resolveInitialSlideKey({ defaultSlideKey, requestedSlideKey, slides }) {
    if (requestedSlideKey && getSlideEntry(slides, requestedSlideKey)) {
      return requestedSlideKey;
    }
    if (defaultSlideKey && getSlideEntry(slides, defaultSlideKey)) {
      return defaultSlideKey;
    }
    return slides[0]?.key ?? null;
  }

  function resolveInitialExperiment({
    availableExperiments,
    defaultExperiment,
    pendingExperiment,
    slideAvailableExperiments,
  }) {
    const candidateExperiments = [
      pendingExperiment,
      typeof defaultExperiment === "string" ? defaultExperiment.trim() : "",
      slideAvailableExperiments[0],
      availableExperiments[0],
    ].filter(Boolean);

    for (const experimentName of candidateExperiments) {
      if (
        slideAvailableExperiments.length === 0 ||
        slideAvailableExperiments.includes(experimentName)
      ) {
        return experimentName;
      }
    }
    return null;
  }

  function getFilteredSlidesFor(slides, filters) {
    return slides.filter((slide) => {
      if (filters.diagnosis && slide.diagnosis !== filters.diagnosis) {
        return false;
      }
      if (filters.infiltration && slide.infiltration !== filters.infiltration) {
        return false;
      }
      return true;
    });
  }

  function willFilterOptionChangeContext({ activeFilters, filterKey, slides, value }) {
    const oppositeFilterKey =
      filterKey === "diagnosis" ? "infiltration" : "diagnosis";
    const oppositeFilterValue = activeFilters[oppositeFilterKey];
    if (!oppositeFilterValue) {
      return false;
    }

    return !slides.some(
      (slide) =>
        slide[filterKey] === value &&
        slide[oppositeFilterKey] === oppositeFilterValue,
    );
  }

  function resolveFilterTransition({ activeFilters, nextFilters, slides }) {
    const requestedFilters = {
      ...activeFilters,
      ...nextFilters,
    };
    const strictMatches = getFilteredSlidesFor(slides, requestedFilters);
    if (strictMatches.length > 0) {
      return {
        filters: requestedFilters,
        matchingSlides: strictMatches,
      };
    }

    const changedKeys = Object.keys(nextFilters);
    if (changedKeys.length !== 1) {
      return {
        filters: requestedFilters,
        matchingSlides: strictMatches,
      };
    }

    const changedKey = changedKeys[0];
    const changedValue = requestedFilters[changedKey];
    if (!changedValue) {
      return {
        filters: requestedFilters,
        matchingSlides: strictMatches,
      };
    }

    const categoryMatches = slides.filter((slide) => slide[changedKey] === changedValue);
    if (categoryMatches.length === 0) {
      return {
        filters: requestedFilters,
        matchingSlides: strictMatches,
      };
    }

    const firstMatch = categoryMatches[0];
    const adjustedFilters = {
      diagnosis:
        changedKey === "diagnosis" ? changedValue : firstMatch.diagnosis,
      infiltration:
        changedKey === "infiltration" ? changedValue : firstMatch.infiltration,
    };
    return {
      filters: adjustedFilters,
      matchingSlides: getFilteredSlidesFor(slides, adjustedFilters),
    };
  }

  return Object.freeze({
    getFilteredSlidesFor,
    getSlideAvailableExperiments,
    getSlideEntry,
    resolveFilterTransition,
    resolveInitialExperiment,
    resolveInitialSlideKey,
    willFilterOptionChangeContext,
  });
})();
