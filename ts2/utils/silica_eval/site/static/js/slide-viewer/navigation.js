const SilicaSlideNavigation = (() => {
  const {
    FILTER_ALL_VALUE,
    SLIDE_QUERY_PARAM,
  } = SilicaSiteConfig;
  const { getFilterDisplayLabel } = SilicaSlideModel;
  const {
    getFilteredSlidesFor: getFilteredSlidesForSelection,
    getSlideAvailableExperiments: getSlideAvailableExperimentsForSelection,
    getSlideEntry: getSlideEntryFromSelection,
    resolveFilterTransition: resolveFilterTransitionForSelection,
    resolveInitialExperiment: resolveInitialExperimentForSelection,
    resolveInitialSlideKey: resolveInitialSlideKeyForSelection,
    willFilterOptionChangeContext: willFilterOptionChangeContextForSelection,
  } = SilicaSlideSelection;

  function populateSlideSelector({ state, els }) {
    const matchingSlides = getFilteredSlides(state);
    const slideSelect = els.slideSelect;
    const previousValue = slideSelect.value;
    const selectedSlideKey = state.pendingSlideKey || state.currentSlideKey;
    slideSelect.replaceChildren();

    if (matchingSlides.length === 0) {
      const option = document.createElement("option");
      option.value = selectedSlideKey || "";
      option.textContent = selectedSlideKey || "No slides";
      slideSelect.appendChild(option);
      slideSelect.disabled = true;
      syncSlideNavigationButtons({ state, els });
      return;
    }

    for (const slide of matchingSlides) {
      const option = document.createElement("option");
      option.value = slide.key;
      option.textContent = slide.label;
      slideSelect.appendChild(option);
    }
    if (
      selectedSlideKey &&
      !matchingSlides.some((slide) => slide.key === selectedSlideKey)
    ) {
      const option = document.createElement("option");
      option.value = selectedSlideKey;
      option.textContent = selectedSlideKey;
      option.disabled = true;
      slideSelect.appendChild(option);
    }

    slideSelect.disabled = false;
    if (selectedSlideKey) {
      slideSelect.value = selectedSlideKey;
      syncSlideNavigationButtons({ state, els });
      return;
    }
    if (matchingSlides.some((slide) => slide.key === previousValue)) {
      slideSelect.value = previousValue;
      syncSlideNavigationButtons({ state, els });
      return;
    }
    slideSelect.value = matchingSlides[0].key;
    syncSlideNavigationButtons({ state, els });
  }

  function syncSlideNavigationButtons({ state, els }) {
    const previousButton = els.previousSlideButton;
    const nextButton = els.nextSlideButton;

    const matchingSlides = getFilteredSlides(state);
    const currentIndex = matchingSlides.findIndex(
      (slide) => slide.key === state.currentSlideKey,
    );
    previousButton.disabled = currentIndex <= 0;
    nextButton.disabled =
      currentIndex === -1 || currentIndex >= matchingSlides.length - 1;
  }

  function populateFilterSelectors({ state, els }) {
    populateFilterSelector({
      state,
      els,
      selectId: "diagnosisSelect",
      filterKey: "diagnosis",
      defaultLabel: "Diagnosis",
    });
    populateFilterSelector({
      state,
      els,
      selectId: "infiltrationSelect",
      filterKey: "infiltration",
      defaultLabel: "Infiltration",
    });
  }

  function populateFilterSelector({ state, els, selectId, filterKey, defaultLabel }) {
    const select = els[selectId];
    select.replaceChildren();

    const placeholderOption = document.createElement("option");
    placeholderOption.value = "";
    placeholderOption.textContent = defaultLabel;
    placeholderOption.hidden = true;
    select.appendChild(placeholderOption);

    const defaultOption = document.createElement("option");
    defaultOption.value = FILTER_ALL_VALUE;
    defaultOption.textContent = "All";
    select.appendChild(defaultOption);

    for (const value of state.filterOptions[filterKey]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = getFilterDisplayLabel(filterKey, value, defaultLabel);
      if (willFilterOptionChangeContext({ state, filterKey, value })) {
        option.classList.add("will-change-slide");
      }
      select.appendChild(option);
    }

    select.value = state.activeFilters[filterKey] || "";
  }

  function willFilterOptionChangeContext({ state, filterKey, value }) {
    return willFilterOptionChangeContextForSelection({
      activeFilters: state.activeFilters,
      filterKey,
      slides: state.slides,
      value,
    });
  }

  function resolveInitialSlideKey({ state, defaultSlideKey, search }) {
    const searchParams = new URLSearchParams(search);
    return resolveInitialSlideKeyForSelection({
      defaultSlideKey,
      requestedSlideKey: searchParams.get(SLIDE_QUERY_PARAM),
      slides: state.slides,
    });
  }

  function getSlideEntry(state, slideKey) {
    return getSlideEntryFromSelection(state.slides, slideKey);
  }

  function getSlideAvailableExperiments(state, slideKey) {
    return getSlideAvailableExperimentsForSelection(state.slides, slideKey);
  }

  function resolveInitialExperiment({ state, defaultExperiment, slideKey }) {
    return resolveInitialExperimentForSelection({
      availableExperiments: state.availableExperiments,
      defaultExperiment,
      pendingExperiment: state.pendingExperiment,
      slideAvailableExperiments: getSlideAvailableExperiments(state, slideKey),
    });
  }

  function ensureValidCurrentExperiment({ state, commit, slideKey }) {
    const slideAvailableExperiments = getSlideAvailableExperiments(state, slideKey);
    if (slideAvailableExperiments.length === 0) {
      if (state.currentExperiment) {
        return state.currentExperiment;
      }
      return null;
    }
    if (state.currentExperiment) {
      return state.currentExperiment;
    }
    const fallbackExperiment = slideAvailableExperiments[0];
    commit({ currentExperiment: fallbackExperiment });
    return fallbackExperiment;
  }

  function populateExperimentSelector({ state, els, commit, slideKey }) {
    const select = els.experimentSelect;

    const slideEntry = getSlideEntry(state, slideKey);
    const slideAvailableExperiments = getSlideAvailableExperiments(state, slideKey);
    const resolvedExperiment = ensureValidCurrentExperiment({ state, commit, slideKey });
    select.replaceChildren();

    if (state.availableExperiments.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Experiment";
      select.appendChild(option);
      select.disabled = true;
      return;
    }

    for (const experimentName of state.availableExperiments) {
      const option = document.createElement("option");
      option.value = experimentName;
      option.textContent = experimentName;
      option.disabled =
        Boolean(slideEntry) && !slideAvailableExperiments.includes(experimentName);
      select.appendChild(option);
    }
    if (
      state.currentExperiment &&
      !state.availableExperiments.includes(state.currentExperiment)
    ) {
      const option = document.createElement("option");
      option.value = state.currentExperiment;
      option.textContent = state.currentExperiment;
      option.disabled = true;
      select.appendChild(option);
    }

    select.disabled = state.availableExperiments.length === 0;
    if (resolvedExperiment !== null) {
      select.value = resolvedExperiment;
    }
  }

  function getFilteredSlidesFor(state, filters) {
    return getFilteredSlidesForSelection(state.slides, filters);
  }

  function getFilteredSlides(state) {
    return getFilteredSlidesFor(state, state.activeFilters);
  }

  function resolveFilterTransition({ state, nextFilters }) {
    return resolveFilterTransitionForSelection({
      activeFilters: state.activeFilters,
      nextFilters,
      slides: state.slides,
    });
  }

  function navigateFilteredSlides({
    state,
    els,
    direction,
    syncUiStateQuery,
    loadSlideFromUi,
  }) {
    const matchingSlides = getFilteredSlides(state);
    const currentIndex = matchingSlides.findIndex(
      (slide) => slide.key === state.currentSlideKey,
    );
    if (currentIndex === -1) {
      syncSlideNavigationButtons({ state, els });
      return;
    }

    const nextIndex = currentIndex + direction;
    if (nextIndex < 0 || nextIndex >= matchingSlides.length) {
      syncSlideNavigationButtons({ state, els });
      return;
    }

    const nextSlideKey = matchingSlides[nextIndex].key;
    els.slideSelect.value = nextSlideKey;
    syncSlideNavigationButtons({ state, els });
    loadSlideFromUi(nextSlideKey, {
      preserveViewport: true,
      useAvailableExperiment: true,
    });
  }

  function applyFilters({
    state,
    els,
    commit,
    nextFilters,
    loadSlideFromUi,
  }) {
    const { filters, matchingSlides } = resolveFilterTransition({ state, nextFilters });
    commit({ activeFilters: filters });
    populateFilterSelectors({ state, els });
    populateSlideSelector({ state, els });
    if (matchingSlides.length === 0) {
      return;
    }

    const preferredSlideKey = matchingSlides.some(
      (slide) => slide.key === state.currentSlideKey,
    )
      ? state.currentSlideKey
      : matchingSlides[0].key;
    els.slideSelect.value = preferredSlideKey;
    if (preferredSlideKey !== state.currentSlideKey) {
      loadSlideFromUi(preferredSlideKey, {
        preserveViewport: true,
        useAvailableExperiment: true,
      });
    }
  }

  return Object.freeze({
    applyFilters,
    ensureValidCurrentExperiment,
    getFilteredSlides,
    getFilteredSlidesFor,
    getSlideAvailableExperiments,
    getSlideEntry,
    navigateFilteredSlides,
    populateExperimentSelector,
    populateFilterSelectors,
    populateSlideSelector,
    resolveFilterTransition,
    resolveInitialExperiment,
    resolveInitialSlideKey,
    syncSlideNavigationButtons,
    willFilterOptionChangeContext,
  });
})();
