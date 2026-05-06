const SilicaUI = (() => {
  const TOPBAR_REVEAL_EVENTS = ["pointerdown", "keydown", "wheel", "touchstart"];
  const STATUS_CLASSES = ["is-working", "is-ready", "is-error"];

  function revealTopbarSelectionLabels(state) {
    if (state.topbarSelectionRevealed) {
      return;
    }
    state.topbarSelectionRevealed = true;
    document.body.classList.add("topbar-selection-revealed");
  }

  function bindTopbarSelectionReveal({ state, triggerSelectors }) {
    const revealOnce = () => {
      revealTopbarSelectionLabels(state);
    };

    for (const selector of triggerSelectors) {
      const element = document.querySelector(selector);
      if (!element) {
        throw new Error(`Missing topbar reveal trigger: ${selector}`);
      }
      element.addEventListener("pointerenter", revealOnce, {
        once: true,
        passive: true,
      });
      element.addEventListener("pointerdown", revealOnce, {
        once: true,
        passive: true,
      });
    }

    for (const eventName of TOPBAR_REVEAL_EVENTS) {
      document.addEventListener(eventName, revealOnce, {
        once: true,
        passive: true,
      });
    }
  }

  function bindResponsivePlacement({
    element,
    mobileSlot,
    desktopParent,
    mediaQueryText,
    anchorLabel,
    onChange,
  }) {
    if (!element || !mobileSlot || !desktopParent) {
      throw new Error(`Missing responsive placement target: ${anchorLabel}`);
    }

    const desktopAnchor = document.createComment(anchorLabel);
    element.after(desktopAnchor);
    const mediaQuery = window.matchMedia(mediaQueryText);
    const syncPlacement = () => {
      const nextParent = mediaQuery.matches ? mobileSlot : desktopParent;
      const nextSibling = mediaQuery.matches ? null : desktopAnchor;
      if (element.parentElement !== nextParent) {
        nextParent.insertBefore(element, nextSibling);
      }
      onChange?.();
    };

    syncPlacement();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", syncPlacement);
    } else {
      mediaQuery.addListener(syncPlacement);
    }
  }

  function bindSidebarCollapse({ state, buttonIds, onChange }) {
    const buttons = buttonIds
      .map((buttonId) => document.getElementById(buttonId))
      .filter(Boolean);
    if (buttons.length === 0) {
      throw new Error("Missing sidebar collapse button");
    }

    const setCollapsed = (isCollapsed) => {
      state.sidebarCollapsed = isCollapsed;
      document.body.classList.toggle("sidebar-collapsed", isCollapsed);
      for (const button of buttons) {
        button.setAttribute("aria-expanded", String(!isCollapsed));
        button.setAttribute(
          "aria-label",
          isCollapsed ? "Expand sidebar" : "Collapse sidebar",
        );
      }
      onChange?.();
    };

    for (const button of buttons) {
      button.addEventListener("click", () => {
        setCollapsed(!state.sidebarCollapsed);
      });
    }
    setCollapsed(state.sidebarCollapsed);
    return { setCollapsed };
  }

  function bindLayoutResize({
    state,
    handleId,
    updateFromPointer,
    onEnd,
    activeKey = "layoutResizeActive",
  }) {
    const resizeHandle = document.getElementById(handleId);
    if (!resizeHandle) {
      throw new Error(`Missing layout resize handle: ${handleId}`);
    }

    resizeHandle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      state[activeKey] = true;
      document.body.classList.add("is-resizing-layout");
      resizeHandle.setPointerCapture(event.pointerId);
      updateFromPointer(event.clientX);
    });

    resizeHandle.addEventListener("pointermove", (event) => {
      if (!state[activeKey]) {
        return;
      }
      updateFromPointer(event.clientX);
    });

    const endResize = (event) => {
      if (!state[activeKey]) {
        return;
      }
      state[activeKey] = false;
      document.body.classList.remove("is-resizing-layout");
      if (resizeHandle.hasPointerCapture(event.pointerId)) {
        resizeHandle.releasePointerCapture(event.pointerId);
      }
      onEnd?.();
    };

    resizeHandle.addEventListener("pointerup", endResize);
    resizeHandle.addEventListener("pointercancel", endResize);
  }

  function setStatusPill({ state, status, pillSelector, statusName, hideMs = 1000 }) {
    if (!["loading", "ready", "error"].includes(status)) {
      throw new Error(`Unknown ${statusName} status: ${status}`);
    }
    const statusPill = document.querySelector(pillSelector);
    if (!statusPill) {
      throw new Error(`Missing status pill: ${pillSelector}`);
    }
    if (state.topbarStatusHideTimer !== null) {
      window.clearTimeout(state.topbarStatusHideTimer);
      state.topbarStatusHideTimer = null;
    }
    state.topbarStatus = status;
    statusPill.classList.toggle("is-working", status === "loading");
    statusPill.classList.toggle("is-ready", status === "ready");
    statusPill.classList.toggle("is-error", status === "error");
    if (status === "ready") {
      state.topbarStatusHideTimer = window.setTimeout(() => {
        statusPill.classList.remove(...STATUS_CLASSES);
        state.topbarStatus = "idle";
        state.topbarStatusHideTimer = null;
      }, hideMs);
    }
  }

  return {
    revealTopbarSelectionLabels,
    bindTopbarSelectionReveal,
    bindResponsivePlacement,
    bindSidebarCollapse,
    bindLayoutResize,
    setStatusPill,
  };
})();
