const SilicaSlideClusterFilter = (() => {
  const { formatInteger } = SilicaSiteUtils;
  const { getAllClusterIds } = SilicaSlideStats;

  function formatClusterFilterValue(clusterIds) {
    return [...clusterIds].sort((left, right) => left - right).join(",");
  }

  function setClusterFilterStatus(statusElement, message) {
    statusElement.textContent = message || "";
  }

  function syncClusterFilterEditor({
    input,
    statusElement,
    cells,
    pendingClusterFilterIds,
    activeClusterFilters,
  }) {
    if (!cells) {
      if (pendingClusterFilterIds === "none") {
        input.value = "none";
        setClusterFilterStatus(statusElement, "Showing no clusters.");
        return;
      }
      if (Array.isArray(pendingClusterFilterIds)) {
        input.value = formatClusterFilterValue(pendingClusterFilterIds);
        setClusterFilterStatus(
          statusElement,
          `Pending ${formatInteger(pendingClusterFilterIds.length)} cluster filter(s).`,
        );
        return;
      }
      input.value = "";
      setClusterFilterStatus(statusElement, "Leave empty for all clusters.");
      return;
    }

    const allClusterIds = getAllClusterIds(cells);
    const selectedClusterIds = Array.from(activeClusterFilters).sort(
      (left, right) => left - right,
    );
    if (selectedClusterIds.length === 0) {
      input.value = "none";
      setClusterFilterStatus(statusElement, "Showing no clusters.");
      return;
    }
    if (
      allClusterIds.length > 0 &&
      allClusterIds.every((clusterId) => activeClusterFilters.has(clusterId))
    ) {
      input.value = "";
      setClusterFilterStatus(
        statusElement,
        `Showing all ${formatInteger(allClusterIds.length)} clusters.`,
      );
      return;
    }

    input.value = formatClusterFilterValue(selectedClusterIds);
    setClusterFilterStatus(
      statusElement,
      `Showing ${formatInteger(selectedClusterIds.length)} of ${formatInteger(
        allClusterIds.length,
      )} clusters.`,
    );
  }

  function resolveClusterFilterInputValue({ rawValue, allClusterIds }) {
    const trimmedValue = rawValue.trim();
    const normalizedValue = trimmedValue.toLowerCase();
    if (!trimmedValue || normalizedValue === "all") {
      return { activeClusterFilters: new Set(allClusterIds), status: null };
    }
    if (normalizedValue === "none") {
      return { activeClusterFilters: new Set(), status: null };
    }

    const requestedClusterIds = trimmedValue
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
      .map((value) => Number.parseInt(value, 10))
      .filter((value) => Number.isInteger(value));
    if (requestedClusterIds.length === 0) {
      return {
        activeClusterFilters: new Set(allClusterIds),
        status: "No valid cluster ids in list. Showing all clusters.",
      };
    }

    const validClusterIds = [...new Set(requestedClusterIds)].filter((clusterId) =>
      allClusterIds.includes(clusterId),
    );
    return {
      activeClusterFilters:
        validClusterIds.length > 0 ? new Set(validClusterIds) : new Set(allClusterIds),
      status:
        validClusterIds.length === 0
          ? "No matching cluster ids in list. Showing all clusters."
          : null,
    };
  }

  function snapshotClusterFilterState({
    cells,
    activeClusterFilters,
    pendingClusterFilterIds,
  }) {
    if (cells) {
      const allClusterIds = getAllClusterIds(cells);
      if (activeClusterFilters.size === 0) {
        return "none";
      }
      if (
        allClusterIds.length > 0 &&
        allClusterIds.every((clusterId) => activeClusterFilters.has(clusterId))
      ) {
        return "all";
      }
      return Array.from(activeClusterFilters).sort((left, right) => left - right);
    }
    if (pendingClusterFilterIds === "all" || pendingClusterFilterIds === "none") {
      return pendingClusterFilterIds;
    }
    if (Array.isArray(pendingClusterFilterIds)) {
      return [...pendingClusterFilterIds].sort((left, right) => left - right);
    }
    return null;
  }

  function resolvePendingClusterFilterPatch({ cells, pendingClusterFilterIds }) {
    if (pendingClusterFilterIds === null) {
      return null;
    }

    const allClusterIds = getAllClusterIds(cells);
    if (pendingClusterFilterIds === "all") {
      return {
        activeClusterFilters: new Set(allClusterIds),
        pendingClusterFilterIds: null,
      };
    }
    if (pendingClusterFilterIds === "none") {
      return {
        activeClusterFilters: new Set(),
        pendingClusterFilterIds: null,
      };
    }

    const validClusterIds = pendingClusterFilterIds.filter((clusterId) =>
      allClusterIds.includes(clusterId),
    );
    return {
      activeClusterFilters:
        validClusterIds.length > 0 ? new Set(validClusterIds) : new Set(allClusterIds),
      pendingClusterFilterIds: null,
    };
  }

  return Object.freeze({
    formatClusterFilterValue,
    resolveClusterFilterInputValue,
    resolvePendingClusterFilterPatch,
    setClusterFilterStatus,
    snapshotClusterFilterState,
    syncClusterFilterEditor,
  });
})();
