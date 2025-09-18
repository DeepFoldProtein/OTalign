from datasets import load_dataset


def load_malidup():
    return load_dataset("DeepFoldProtein/malidup-dataset", split="test")


def load_malisam():
    return load_dataset("DeepFoldProtein/malisam-dataset", split="test")


def load_sabmark(name: str = "all"):
    return load_dataset("DeepFoldProtein/SABmark-dataset", name=name, split="test")
