"""Mechanical token IDs -> original random-weight model -> one backward step."""
from pathlib import Path

import torch

from h2lm import H2LM
from h2lm.config import H2LMConfig, ModelConfig, VisionConfig
from h2lm.tokenization import H2Tokenizer, TokenizerConfig, train_tokenizer


def test_real_token_ids_multimodal_backward(tmp_path):
    root = Path(__file__).resolve().parents[1]
    train_tokenizer(TokenizerConfig(), root / "data/tokenizer_sample/manifest.json",
                    tmp_path / "tokenizer")
    tokenizer = H2Tokenizer.load(tmp_path / "tokenizer")
    config = H2LMConfig(
        name="m1-integration", model=ModelConfig(tokenizer.vocab_size, 128, 32, 1, 4),
        vision=VisionConfig(32, 16, 3, 32, 1, 4),
    )
    tokenizer.require_model_vocab(config.model.vocab_size)
    torch.manual_seed(29)
    model = H2LM(config)
    ids = torch.tensor([tokenizer.encode("Điều 12. Mẫu tiếng Việt.", bos=True, eos=True)])
    output = model(ids, torch.randn(1, 3, 32, 32), labels=ids)
    assert output["logits"].shape == (1, ids.shape[1], tokenizer.vocab_size)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert torch.isfinite(model.token_embedding.weight.grad).all()
    assert torch.isfinite(model.vision.patch_embed.weight.grad).all()
    assert model.vision.patch_embed.weight.grad.abs().sum() > 0
