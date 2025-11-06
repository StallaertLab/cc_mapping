"""Integration tests for GMMThresholding workflows.

This module tests end-to-end workflows combining multiple features:
fit → categorize → plot, with various combinations of automatic/manual
thresholding and label collapsing.

Priority 3 - To be implemented after Priority 1 & 2 tests pass.
"""

import pytest
from src.cc_mapping.thresholding import GMMThresholding

# TODO: Add tests for:
# - Complete workflow: fit → automatic categorize → plot
# - Complete workflow: fit → manual categorize → plot
# - Complete workflow: fit → collapsed labels → plot
# - Switching between automatic and manual thresholding
# - Re-categorizing with different parameters
# - State consistency across operations
# - Real-world usage patterns
