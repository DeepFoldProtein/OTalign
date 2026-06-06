from functools import partial

from .ankh_adaptor import build_ankh_adaptor
from .ankhcl_adaptor import build_ankhcl_adaptor
from .esm_adaptor import build_esm_adaptor
from .esmc_adaptor import build_esmc_adaptor
from .proteinglm_int4_adaptor import build_proteinglm_int4_adaptor
from .t5_adaptor import build_prott5_adaptor


def get_plm_adaptor_and_configs(name: str, for_masked_lm: bool = False):
    MODEL_REGISTRY = {
        "ESM2_36_3B": {
            "builder": partial(build_esm_adaptor, "facebook/esm2_t36_3B_UR50D", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-2 (36L-3B)",
        },
        "ESM2_33_650M": {
            "builder": partial(build_esm_adaptor, "facebook/esm2_t33_650M_UR50D", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-2 (33L-650M)",
        },
        "ESM2_30_150M": {
            "builder": partial(build_esm_adaptor, "facebook/esm2_t30_150M_UR50D", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-2 (30L-150M)",
        },
        "ESM2_12_35M": {
            "builder": partial(build_esm_adaptor, "facebook/esm2_t12_35M_UR50D", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-2 (12L-35M)",
        },
        "ESM2_6_8M": {
            "builder": partial(build_esm_adaptor, "facebook/esm2_t6_8M_UR50D", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-2 (6L-8M)",
        },
        "ESM1b_33_650M": {
            "builder": partial(build_esm_adaptor, "facebook/esm1b_t33_650M_UR50S", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-1b",
        },
        "ESMC_300M": {
            "builder": partial(build_esmc_adaptor, "300m", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-C (300M)",
        },
        "ESMC_600M": {
            "builder": partial(build_esmc_adaptor, "600m", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-C (600M)",
        },
        "ESMC_6B": {
            "builder": partial(build_esmc_adaptor, "6b", for_masked_lm=for_masked_lm),
            "policy": "drop_first_last_active",
            "adaptor_name": "ESM-C (6B)",
        },
        "AnkhCL": {
            "builder": partial(build_ankhcl_adaptor, for_masked_lm=for_masked_lm),
            "policy": "drop_last_active",
            "adaptor_name": "AnkhCL",
        },
        "Ankh_Base": {
            "builder": partial(build_ankh_adaptor, model_name="ElnaggarLab/ankh-base", for_masked_lm=for_masked_lm),
            "policy": "drop_last_active",
            "adaptor_name": "Ankh-Base",
        },
        "Ankh_Large": {
            "builder": partial(build_ankh_adaptor, model_name="ElnaggarLab/ankh-large", for_masked_lm=for_masked_lm),
            "policy": "drop_last_active",
            "adaptor_name": "Ankh-Large",
        },
        "Ankh3_Large": {
            "builder": partial(build_ankh_adaptor, model_name="ElnaggarLab/ankh3-large", for_masked_lm=for_masked_lm),
            "policy": "drop_last_active",
            "adaptor_name": "Ankh3-Large",
        },
        "Ankh3_XL": {
            "builder": partial(build_ankh_adaptor, model_name="ElnaggarLab/ankh3-xl", for_masked_lm=for_masked_lm),
            "policy": "drop_last_active",
            "adaptor_name": "Ankh3-XL",
        },
        "ProtT5_XL_UniRef50": {
            "builder": partial(build_prott5_adaptor, "Rostlab/prot_t5_xl_uniref50", for_masked_lm=for_masked_lm),
            "policy": "drop_last_active",
            "adaptor_name": "ProtT5_XL_UniRef50",
        },
        "ProteinGLM_100B_INT4": {
            "builder": partial(build_proteinglm_int4_adaptor, "Bo1015/proteinglm-100b-int4", for_masked_lm=for_masked_lm),
            "policy": "drop_last_active",
            "adaptor_name": "ProteinGLM-100B",
        },
    }

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Not valid model name: '{name}'. Available models: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[name]
    adaptor = config["builder"]()

    return adaptor, config["policy"], config["adaptor_name"]
