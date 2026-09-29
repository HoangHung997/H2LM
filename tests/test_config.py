from pathlib import Path

from h2lm import load_config


def test_dev_yaml_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/model/h2lm_v1_dev.yaml")

    assert config.name == "h2lm-v1-dev"
    assert config.model.vocab_size == 8192
    assert config.vision.image_size == 512
