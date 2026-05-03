"""Reusable filters for research pipelines."""

from .quantile_filter import (
    QuantileRange,
    assign_equal_frequency_quantile_labels,
    filter_by_quantile_range,
    maybe_filter_by_quantile_range,
    maybe_parse_quantile_range,
    normalize_bucket_label,
    parse_quantile_range,
    suppress_entries_outside_quantile_range,
)

__all__ = [
    "QuantileRange",
    "assign_equal_frequency_quantile_labels",
    "filter_by_quantile_range",
    "maybe_filter_by_quantile_range",
    "maybe_parse_quantile_range",
    "normalize_bucket_label",
    "parse_quantile_range",
    "suppress_entries_outside_quantile_range",
]
