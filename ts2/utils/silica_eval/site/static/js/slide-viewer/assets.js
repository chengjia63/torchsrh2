const SilicaSlideAssets = (() => {
  function portalAssetUrl({ slideKey, experimentName, relativePath }) {
    return new URL(
      relativePath,
      `${window.location.origin}${window.SILICA_SLIDE_VIEWER.portalAssetBaseUrl}/${encodeURIComponent(
        experimentName,
      )}/${encodeURIComponent(slideKey)}/`,
    ).toString();
  }

  function dziAssetUrl({ slideKey, relativePath }) {
    return new URL(
      relativePath,
      `${window.location.origin}${window.SILICA_SLIDE_VIEWER.dziAssetBaseUrl}/${encodeURIComponent(slideKey)}/`,
    ).toString();
  }

  return Object.freeze({
    dziAssetUrl,
    portalAssetUrl,
  });
})();
