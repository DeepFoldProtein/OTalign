from .ankhcl_adaptor import build_ankhcl_adaptor
from .esm_adaptor import build_esm_adaptor
from .t5_adaptor import build_prott5_adaptor


REGISTRY = {
    "ankhcl": build_ankhcl_adaptor,
    "esm2": build_esm_adaptor,
    "prott5": build_prott5_adaptor,
}
