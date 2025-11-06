"""Configuration for the API reference documentation."""


def _get_guide(*refs):
    """Get the rst to refer to user guide."""
    if len(refs) == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif len(refs) == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        ref_desc = ", ".join(f":ref:`{ref}`" for ref in refs[:-1])
        ref_desc += f", and :ref:`{refs[-1]}` sections"

    return f"**User guide.** See the {ref_desc} for further details."


API_REFERENCE = {
    "cc_mapping.thresholding": {
        "short_summary": "Thresholding tools for cell cycle analysis.",
        "description": "Tools for Gaussian Mixture Model-based thresholding to identify cell cycle phases.",
        "sections": [
            {
                "title": "Classes",
                "autosummary": [
                    "GMMThresholding",
                    "SequentialGMM",
                ],
            },
        ],
    },
    "cc_mapping.utils": {
        "short_summary": "General utility functions.",
        "description": "General-purpose utility functions for data manipulation and analysis.",
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "create_boolean_label_combination",
                ],
            },
        ],
    },
}
