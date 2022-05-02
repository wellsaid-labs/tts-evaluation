from functools import partial

from run.data._loader import structures as struc
from run.data._loader.utils import wsl_gcs_dataset_loader

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

Dia = struc.Dialect


def make(
    label: str,
    name: str,
    dialect: struc.Dialect = Dia.EN_US,
    style: struc.Style = struc.Style.RND,
) -> struc.Speaker:
    return struc.Speaker(label, style, dialect, name, label)


RND__AVA_M_V10_COVERAGE = make("rnd__ava_m_v10_coverage", "Ava M (v10 R&D)")
RND__GENEVIEVE_M_V10_COVERAGE = make("rnd__genevieve_m_v10_coverage", "Genevieve M (v10 R&D)")
RND__JARVIS_H_V10_COVERAGE = make("rnd__jarvis_h_v10_coverage", "Jarvis H (v10 R&D)")
RND__WADE_C_V10_COVERAGE = make("rnd__wade_c_v10_coverage", "Wade C (v10 R&D)")

_rnd_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
RND_DATASETS = {s: partial(wsl_gcs_dataset_loader, speaker=s) for s in _rnd_speakers}
