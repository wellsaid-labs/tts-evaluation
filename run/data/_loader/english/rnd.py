from functools import partial

from run.data._loader import structures as struc
from run.data._loader.english.wsl import AVA_M, GENEVIEVE_M, JARVIS_H, WADE_C
from run.data._loader.utils import wsl_gcs_dataset_loader

Dia = struc.Dialect


def make(
    label: str,
    name: str,
    gcs_dir: str,
    dialect: struc.Dialect = Dia.EN_US,
    style: struc.Style = struc.Style.RND,
) -> struc.Speaker:
    return struc.Speaker(label, style, dialect, name, gcs_dir)


AVA_M__RND_V10_COVERAGE = make(
    AVA_M.label, "Ava M (v10 R&D)", "rnd__ava_m_v10_coverage", AVA_M.dialect
)
GENEVIEVE_M__RND_V10_COVERAGE = make(
    GENEVIEVE_M.label, "Genevieve M (v10 R&D)", "rnd__genevieve_m_v10_coverage", GENEVIEVE_M.dialect
)
JARVIS_H__RND_V10_COVERAGE = make(
    JARVIS_H.label, "Jarvis H (v10 R&D)", "rnd__jarvis_h_v10_coverage", JARVIS_H.dialect
)
WADE_C__RND_V10_COVERAGE = make(
    WADE_C.label, "Wade C (v10 R&D)", "rnd__wade_c_v10_coverage", WADE_C.dialect
)

_rnd_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
RND_DATASETS = {s: partial(wsl_gcs_dataset_loader, speaker=s) for s in _rnd_speakers}
