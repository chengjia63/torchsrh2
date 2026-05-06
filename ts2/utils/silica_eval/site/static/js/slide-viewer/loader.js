const SilicaSlideLoader = (() => {
  const { fetchJson } = SilicaSiteUtils;
  const { SLIDE_CELLS_PATH } = SilicaSiteConfig;

  async function loadSlideIndex() {
    return fetchJson(window.SILICA_SLIDE_VIEWER.slidesUrl);
  }

  async function loadSlidePayloads({ experimentName, slideKey }) {
    const manifest = await fetchJson(
      SilicaSlideAssets.portalAssetUrl({
        experimentName,
        relativePath: "slide_manifest.json",
        slideKey,
      }),
    );
    const cells = await fetchJson(
      SilicaSlideAssets.portalAssetUrl({
        experimentName,
        relativePath: SLIDE_CELLS_PATH,
        slideKey,
      }),
    );
    return { cells, manifest };
  }

  return Object.freeze({
    loadSlideIndex,
    loadSlidePayloads,
  });
})();
