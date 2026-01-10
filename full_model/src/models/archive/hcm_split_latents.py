"""
DEPRECATED MODULE - MOVED TO ARCHIVE
=====================================

This module has been moved to:
    full_model/src/models/archive/hcm_split_latents.py

REASON: The two-stage estimation approach causes 15-50% ATTENUATION BIAS
on latent variable effect estimates.

USE INSTEAD:
    - models/hcm_basic/model.py (single LV, simultaneous ICLV)
    - models/hcm_full/model.py (4 LVs, simultaneous ICLV)
    - models/iclv/model.py (2 LVs, simultaneous ICLV)

These models use simultaneous estimation which eliminates attenuation bias.

If you still need the deprecated two-stage approach for comparison purposes,
import from the archive:
    from full_model.src.models.archive.hcm_split_latents import *
"""

raise ImportError(
    "hcm_split_latents has been DEPRECATED due to 15-50% attenuation bias. "
    "Use models/hcm_basic/model.py, models/hcm_full/model.py, or models/iclv/model.py instead. "
    "For the archived version, import from: full_model.src.models.archive.hcm_split_latents"
)
