from functools import partial

from .ankhcl_adaptor import build_ankhcl_adaptor
from .esm_adaptor import build_esm_adaptor
from .t5_adaptor import build_prott5_adaptor


MODEL_REGISTRY = {
    "ESM2": {
        "builder": partial(build_esm_adaptor, "facebook/esm2_t33_650M_UR50D"),
        "policy": "drop_first_last_active",
        "adaptor_name": "ESM-2",
    },
    "ESM1b": {
        "builder": partial(build_esm_adaptor, "facebook/esm1b_t33_650M_UR50S"),
        "policy": "drop_first_last_active",
        "adaptor_name": "ESM-1b",
    },
    "AnkhCL": {
        "builder": build_ankhcl_adaptor,
        "policy": "drop_last_active",
        "adaptor_name": "AnkhCL",
    },
    "ProtT5": {
        "builder": partial(build_prott5_adaptor, "Rostlab/prot_t5_xl_uniref50"),
        "policy": "drop_last_active",
        "adaptor_name": "ProtT5",
    },
}


def get_plm_adaptor_and_configs(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Not valid model name: '{name}'. Available models: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[name]
    adaptor = config["builder"]()

    return adaptor, config["policy"], config["adaptor_name"]
