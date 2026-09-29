from __future__ import annotations

import argparse

import torch

from h2lm import H2LM, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one H2LM random-weight training step.")
    parser.add_argument(
        "--config",
        default="configs/model/h2lm_v1_dev.yaml",
        help="Path to YAML model config.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cuda or cpu.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    model = H2LM(config).to(device)
    model.train()

    batch_size = 1
    text_len = min(64, config.model.max_text_tokens)

    input_ids = torch.randint(
        0,
        config.model.vocab_size,
        (batch_size, text_len),
        device=device,
    )
    images = torch.randn(
        batch_size,
        config.vision.in_channels,
        config.vision.image_size,
        config.vision.image_size,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model(input_ids=input_ids, pixel_values=images, labels=input_ids)
    loss = output["loss"]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"model={config.name}")
    print(f"device={device}")
    print(f"parameters={model.parameter_count():,}")
    print(f"vision_tokens={output['vision_tokens']}")
    print(f"loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
