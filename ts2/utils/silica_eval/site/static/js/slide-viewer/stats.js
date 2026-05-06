const SilicaSlideStats = (() => {
  function meanTumorScoreForIndices(cells, indices) {
    if (!indices.length) {
      return null;
    }

    let tumorScoreSum = 0;
    for (const index of indices) {
      tumorScoreSum += cells.tumor_score[index];
    }
    return tumorScoreSum / indices.length;
  }

  function getClusterCounts(cells, indices = null) {
    const counts = new Map();
    const sourceIndices =
      indices ?? Array.from({ length: cells.dominant_cluster.length }, (_, index) => index);
    for (const index of sourceIndices) {
      const cluster = cells.dominant_cluster[index];
      counts.set(cluster, (counts.get(cluster) || 0) + 1);
    }
    return counts;
  }

  function getSortedClustersByCount(cells) {
    return Array.from(getClusterCounts(cells).entries()).sort((left, right) => {
      if (right[1] !== left[1]) {
        return right[1] - left[1];
      }
      return left[0] - right[0];
    });
  }

  function getAllClusterIds(cells) {
    return getSortedClustersByCount(cells).map(([cluster]) => cluster);
  }

  function getTopCluster(cells, indices) {
    const clusterCounts = getClusterCounts(cells, indices);
    if (!clusterCounts.size) {
      return null;
    }

    let topCluster = null;
    let topClusterCount = -1;
    for (const [cluster, count] of clusterCounts.entries()) {
      if (count > topClusterCount) {
        topCluster = cluster;
        topClusterCount = count;
      }
    }

    return {
      cluster: topCluster,
      count: topClusterCount,
    };
  }

  return Object.freeze({
    getAllClusterIds,
    getClusterCounts,
    getSortedClustersByCount,
    getTopCluster,
    meanTumorScoreForIndices,
  });
})();
