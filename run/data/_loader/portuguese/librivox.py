from functools import partial

from run.data._loader.data_structures import make_pt_speaker
from run.data._loader.utils import Speaker, wsl_gcs_dataset_loader

RND__LIBRIVOX__FELIPE_PT = make_pt_speaker(
    "rnd__librivox__felipe_pt", "Felipe (R & D Portuguese Voice)"
)
RND__LIBRIVOX__LENI_PT = make_pt_speaker("rnd__librivox__leni_pt", "Leni (R & D Portuguese Voice)")
RND__LIBRIVOX__MIRAMONTES_PT = make_pt_speaker(
    "rnd__librivox__miramontes_pt", "Miramontes (R & D Portuguese Voice)"
)
RND__LIBRIVOX__SANDRALUNA_PT = make_pt_speaker(
    "rnd__librivox__sandraluna_pt", "Sandra Luna (R & D Portuguese Voice)"
)

_librivox_speakers = [s for s in locals().values() if isinstance(s, Speaker)]
LIBRIVOX_DATASETS = {
    s: partial(wsl_gcs_dataset_loader, speaker=s, prefix=str(s.language.value).lower())
    for s in _librivox_speakers
}
