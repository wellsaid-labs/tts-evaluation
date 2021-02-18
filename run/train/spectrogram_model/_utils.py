from hparams import HParam, configurable

import lib


@configurable
def set_seed(seed=HParam()):
    lib.environment.set_seed(seed)
