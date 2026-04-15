from typing import Dict, List, Sequence, Tuple

from ts2.utils.tailwind import TC


DISPLAY_NAME_BY_EXP: Dict[str, str] = {
    "a2706135_dinov2": "DINOv2 Meta",
    "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0": "DINOv2 lr4e-3",
    "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0": "Silica FullIm iBOT lr4e-3",
    "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0": "DINOv2 lr4e-3 RmBg",
    "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1": "Silica Inside iBOT lr4e-3",
    "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0": "Silica FullIm iBOT lr1e-3",
    "4fb55301_Apr09-01-59-24_sd1000_nomaskobw_lr54_tune0": "Silica FullIM iBOT lr5e-4",
    "326a6384_Apr10-15-07-23_sd1000_nomaskobw_lr14_tune0": "Silica FullIM iBOT lr1e-4",
    "10d41c43_Apr11-02-05-16_sd1000_nomaskobw_lr23_tune0": "Silica FullIM iBOT lr2e-3",
    "716f4772_Apr12-03-21-26_sd1000_maskobw_lr13_tune1": "Silica Inside iBOT lr1e-3",
    "28d7879f_Apr13-02-20-13_sd1000_maskobw_lr54_tune1": "Silica Inside iBOT lr5e-4",
}


def build_display_name_color_range(
    display_names: Sequence[str], shade: int = 5
) -> Tuple[List[str], List[str]]:
    color_domain = sorted(dict.fromkeys(display_names), key=str.casefold)
    if not color_domain:
        raise ValueError("Expected at least one display name to build a color range")
    return color_domain, TC()(nc=len(color_domain), s=shade)
