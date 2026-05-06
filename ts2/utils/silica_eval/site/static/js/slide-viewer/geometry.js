const SilicaSlideGeometry = (() => {
  const { clampNumber } = SilicaSiteUtils;

  function normalizeScreenRect(rect) {
    const left = Math.min(rect.x, rect.x + rect.width);
    const right = Math.max(rect.x, rect.x + rect.width);
    const top = Math.min(rect.y, rect.y + rect.height);
    const bottom = Math.max(rect.y, rect.y + rect.height);
    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    };
  }

  function ensureMinimumScreenRectSize(rect, centerPoint, minSize) {
    const normalized = normalizeScreenRect(rect);
    const width = Math.max(normalized.width, minSize);
    const height = Math.max(normalized.height, minSize);
    return {
      x: centerPoint.x - width / 2,
      y: centerPoint.y - height / 2,
      width,
      height,
    };
  }

  function expandImageRectForScreenMargin({
    imageHeight,
    imageRect,
    imageWidth,
    overlayHeight,
    overlayWidth,
    screenMarginPx,
  }) {
    const safeOverlayWidth = Math.max(1, overlayWidth);
    const safeOverlayHeight = Math.max(1, overlayHeight);
    const imageMarginX = (imageRect.width / safeOverlayWidth) * screenMarginPx;
    const imageMarginY = (imageRect.height / safeOverlayHeight) * screenMarginPx;
    const x = clampNumber(imageRect.x - imageMarginX, 0, imageWidth);
    const y = clampNumber(imageRect.y - imageMarginY, 0, imageHeight);
    const maxX = clampNumber(imageRect.x + imageRect.width + imageMarginX, 0, imageWidth);
    const maxY = clampNumber(imageRect.y + imageRect.height + imageMarginY, 0, imageHeight);
    return {
      x,
      y,
      width: Math.max(0, maxX - x),
      height: Math.max(0, maxY - y),
    };
  }

  function buildPartitionBuckets(points, bucketSize) {
    const buckets = new Map();
    for (let index = 0; index < points.length; index += 1) {
      const point = points[index];
      const bucketX = Math.floor(point.x / bucketSize);
      const bucketY = Math.floor(point.y / bucketSize);
      const bucketKey = `${bucketX},${bucketY}`;
      const bucket = buckets.get(bucketKey);
      if (bucket) {
        bucket.push(index);
      } else {
        buckets.set(bucketKey, [index]);
      }
    }
    return buckets;
  }

  function findNearestPartitionPoint(sampleX, sampleY, points, buckets, bucketSize) {
    const baseBucketX = Math.floor(sampleX / bucketSize);
    const baseBucketY = Math.floor(sampleY / bucketSize);
    let bestPoint = null;
    let bestDistanceSquared = Number.POSITIVE_INFINITY;
    const maxRing = 64;

    for (let ring = 0; ring <= maxRing; ring += 1) {
      const minBucketX = baseBucketX - ring;
      const maxBucketX = baseBucketX + ring;
      const minBucketY = baseBucketY - ring;
      const maxBucketY = baseBucketY + ring;

      for (let bucketX = minBucketX; bucketX <= maxBucketX; bucketX += 1) {
        for (let bucketY = minBucketY; bucketY <= maxBucketY; bucketY += 1) {
          if (
            ring > 0 &&
            bucketX > minBucketX &&
            bucketX < maxBucketX &&
            bucketY > minBucketY &&
            bucketY < maxBucketY
          ) {
            continue;
          }
          const bucket = buckets.get(`${bucketX},${bucketY}`);
          if (!bucket) {
            continue;
          }
          for (const pointIndex of bucket) {
            const point = points[pointIndex];
            const dx = point.x - sampleX;
            const dy = point.y - sampleY;
            const distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < bestDistanceSquared) {
              bestDistanceSquared = distanceSquared;
              bestPoint = point;
            }
          }
        }
      }

      if (bestPoint) {
        const maxGuaranteedDistance = Math.max(0, ring) * bucketSize;
        if (bestDistanceSquared <= maxGuaranteedDistance * maxGuaranteedDistance) {
          break;
        }
      }
    }

    return bestPoint;
  }

  return Object.freeze({
    buildPartitionBuckets,
    ensureMinimumScreenRectSize,
    expandImageRectForScreenMargin,
    findNearestPartitionPoint,
    normalizeScreenRect,
  });
})();
