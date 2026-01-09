"""
Item Detection Utilities
========================
Centralized functions for detecting Likert items by domain/construct.

This module provides utilities for mapping between:
- Unified item names (patriotism_1, secularism_1, etc.)
- Internal latent variable names (pat_blind, pat_constructive, sec_dl, sec_fp)
- Subconstruct item ranges
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd

# Item ranges for each domain
PATRIOTISM_RANGE = (1, 20)
SECULARISM_RANGE = (1, 25)

# Subconstruct mappings: domain -> {lv_name: (start, end)}
SUBCONSTRUCT_MAP = {
    'patriotism': {
        'pat_blind': (1, 10),
        'pat_constructive': (11, 20),
    },
    'secularism': {
        'sec_dl': (1, 15),
        'sec_fp': (16, 25),
    }
}

# Reverse mapping: lv_name -> (domain, lv_name)
LV_TO_DOMAIN = {
    'pat_blind': ('patriotism', 'pat_blind'),
    'pat_constructive': ('patriotism', 'pat_constructive'),
    'sec_dl': ('secularism', 'sec_dl'),
    'sec_fp': ('secularism', 'sec_fp'),
}

# Reverse item indices (1-based, overall domain numbering)
REVERSE_ITEMS = {
    'patriotism': [5, 7, 12],
    'secularism': [1, 13],
}


def get_domain_items(df: pd.DataFrame, domain: str) -> List[str]:
    """Get all items for a domain (patriotism or secularism).

    Args:
        df: DataFrame with item columns
        domain: Either 'patriotism' or 'secularism'

    Returns:
        List of column names matching the domain pattern
    """
    prefix = f"{domain}_"
    return sorted([c for c in df.columns
                   if c.startswith(prefix) and c.split('_')[-1].isdigit()],
                  key=lambda x: int(x.split('_')[-1]))


def get_subconstruct_items(df: pd.DataFrame, domain: str, subconstruct: str) -> List[str]:
    """Get items for a specific subconstruct.

    Args:
        df: DataFrame with item columns
        domain: Either 'patriotism' or 'secularism'
        subconstruct: Internal LV name (pat_blind, pat_constructive, sec_dl, sec_fp)

    Returns:
        List of column names for items in the specified subconstruct
    """
    if domain not in SUBCONSTRUCT_MAP:
        return []
    if subconstruct not in SUBCONSTRUCT_MAP[domain]:
        return []

    start, end = SUBCONSTRUCT_MAP[domain][subconstruct]
    return [f"{domain}_{i}" for i in range(start, end + 1)
            if f"{domain}_{i}" in df.columns]


def get_lv_items(df: pd.DataFrame, lv_name: str) -> List[str]:
    """Get items for a latent variable by internal name.

    Args:
        df: DataFrame with item columns
        lv_name: Internal LV name (pat_blind, pat_constructive, sec_dl, sec_fp)

    Returns:
        List of column names for items measuring the specified LV
    """
    if lv_name in LV_TO_DOMAIN:
        domain, subconst = LV_TO_DOMAIN[lv_name]
        return get_subconstruct_items(df, domain, subconst)
    return []


def get_all_lv_items(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get items for all latent variables.

    Args:
        df: DataFrame with item columns

    Returns:
        Dictionary mapping LV names to their item column names
    """
    return {
        lv_name: get_lv_items(df, lv_name)
        for lv_name in LV_TO_DOMAIN.keys()
    }


def is_reverse_item(domain: str, item_number: int) -> bool:
    """Check if an item is reverse-coded.

    Args:
        domain: Either 'patriotism' or 'secularism'
        item_number: 1-based item index within the domain

    Returns:
        True if the item is reverse-coded
    """
    return item_number in REVERSE_ITEMS.get(domain, [])


def get_reverse_items(domain: str) -> List[int]:
    """Get list of reverse-coded item indices for a domain.

    Args:
        domain: Either 'patriotism' or 'secularism'

    Returns:
        List of 1-based item indices that are reverse-coded
    """
    return REVERSE_ITEMS.get(domain, [])


def item_to_lv(item_name: str) -> Optional[str]:
    """Map an item name to its latent variable.

    Args:
        item_name: Item column name (e.g., 'patriotism_5')

    Returns:
        Internal LV name or None if not recognized
    """
    if not item_name or '_' not in item_name:
        return None

    parts = item_name.rsplit('_', 1)
    if len(parts) != 2:
        return None

    domain, num_str = parts
    try:
        item_num = int(num_str)
    except ValueError:
        return None

    if domain not in SUBCONSTRUCT_MAP:
        return None

    for lv_name, (start, end) in SUBCONSTRUCT_MAP[domain].items():
        if start <= item_num <= end:
            return lv_name

    return None


# Legacy support for old prefix-based detection
def get_items_by_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
    """Legacy function to get items by old-style prefix.

    Maps old prefixes (pat_blind_, pat_constructive_, sec_dl_, sec_fp_)
    to new unified naming.

    Args:
        df: DataFrame with item columns
        prefix: Old-style prefix (e.g., 'pat_blind_')

    Returns:
        List of column names (using new naming) or empty list
    """
    prefix_to_lv = {
        'pat_blind_': 'pat_blind',
        'pat_constructive_': 'pat_constructive',
        'sec_dl_': 'sec_dl',
        'sec_fp_': 'sec_fp',
    }

    lv_name = prefix_to_lv.get(prefix)
    if lv_name:
        return get_lv_items(df, lv_name)

    # Fallback: try direct prefix matching (for backward compatibility)
    return sorted([c for c in df.columns
                   if c.startswith(prefix) and c[-1].isdigit()],
                  key=lambda x: int(x.split('_')[-1]))
