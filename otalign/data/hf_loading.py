from datasets import load_dataset


def load_malidup(split: str = "test"):
    return load_dataset("DeepFoldProtein/malidup-dataset", split=split)


def load_malisam(split: str = "test"):
    return load_dataset("DeepFoldProtein/malisam-dataset", split=split)


def load_sabmark(split: str = "test"):
    return load_dataset("DeepFoldProtein/SABmark-dataset", split=split)
