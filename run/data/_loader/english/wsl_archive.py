import logging

from run.data._loader import structures as struc

logger = logging.getLogger(__name__)


LINCOLN__CUSTOM = struc.Speaker(
    "Lincoln_custom_ma",
    struc.Style.OTHER,
    struc.Dialect.EN_UNKNOWNN,
    "Lincoln (Custom)",
    "Lincoln_custom_ma",
)
JOSIE__CUSTOM = struc.Speaker(
    "Josie_Custom", struc.Style.OTHER, struc.Dialect.EN_NZ, "Josie (Custom)", "Josie_Custom"
)
JOSIE__CUSTOM__MANUAL_POST = struc.Speaker(
    "Josie_Custom",
    struc.Style.OTHER,
    struc.Dialect.EN_NZ,
    "Josie (Custom, Loudness Standardized)",
    "Josie_Custom_Loudnorm",
    post=True,
)

_deprecated_metadata = {
    # NOTE: This dataset isn't available for commercial use. This was a project commissioned by
    # iHeartRadio for a proof of concept. It was never intended to be publicized.
    (
        "Sean Hannity",
        struc.Speaker("Sean Hannity", struc.Style.OTHER, None),  # type: ignore
        "https://drive.google.com/uc?export=download&id=1YHX6yl1kX7lQguxSs4sJ1FPrAS9NZ8O4",
        "Sean Hannity.tar.gz",
        False,
    ),
    # NOTE: This dataset was decommissioned due to poor quality.
    (
        "Nadine Nagamatsu",
        struc.Speaker("Nadine Nagamatsu", struc.Style.OG_NARR, None),  # type: ignore
        "https://drive.google.com/uc?export=download&id=1fwW6oV7x3QYImSfG811vhfjp8jKXVMGZ",
        "Nadine Nagamatsu.tar.gz",
        False,
    ),
    # NOTE: The datasets below are custom voices.
    (
        "Lincoln_custom_ma",
        LINCOLN__CUSTOM,
        "https://drive.google.com/uc?export=download&id=1NJkVrPyxiNLKhc1Pj-ssCFhx_Mxzervf",
        "Lincoln_custom_ma.tar.gz",
        True,
    ),
    (
        "Josie_Custom",
        JOSIE__CUSTOM,
        "https://drive.google.com/uc?export=download&id=1KPPjVMgCWCf-efkZBiCivbpiIt5z3LcG",
        "Josie_Custom.tar.gz",
        True,
    ),
    (
        "Josie_Custom_Loudnorm",
        JOSIE__CUSTOM__MANUAL_POST,
        "https://drive.google.com/uc?export=download&id=1CeLacT0Ys6jiroJPH0U8aO0GaKemg0vK",
        "Josie_Custom_Loudnorm.tar.gz",
        True,
    ),
}
