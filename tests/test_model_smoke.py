import torch

from h2lm.config import H2LMConfig, ModelConfig, VisionConfig
from h2lm.modeling.h2lm import H2LM


def tiny_config() -> H2LMConfig:
    return H2LMConfig(
        name="test",
        model=ModelConfig(
            vocab_size=128,
            max_text_tokens=32,
            d_model=64,
            num_layers=2,
            num_heads=4,
        ),
        vision=VisionConfig(
            image_size=64,
            patch_size=16,
            in_channels=3,
            d_vision=64,
            num_layers=2,
            num_heads=4,
        ),
    )


def test_multimodal_forward_and_loss() -> None:
    model = H2LM(tiny_config())
    input_ids = torch.randint(0, 128, (2, 12))
    images = torch.randn(2, 3, 64, 64)

    output = model(input_ids=input_ids, pixel_values=images, labels=input_ids)

    assert output["logits"].shape == (2, 12, 128)
    assert output["vision_tokens"] == 16
    assert torch.isfinite(output["loss"])


def test_text_only_forward() -> None:
    model = H2LM(tiny_config())
    input_ids = torch.randint(0, 128, (1, 8))

    output = model(input_ids=input_ids)

    assert output["logits"].shape == (1, 8, 128)
    assert output["vision_tokens"] == 0
