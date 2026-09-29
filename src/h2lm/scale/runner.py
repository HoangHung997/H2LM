"""Finite 1B scale experiment. A count/short pilot is not foundation-model pretraining."""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ScaleConfig, tiny_config
from .fixture import decode, encode, example
from .model import build, parameter_report
from .optim import AuditedSGD
from .storage import (
    get_latest,
    index_at,
    load_into,
    save,
    set_latest,
    source_hash,
    write_json,
)


@torch.no_grad()
def predict(model, fixture: dict, dtype: torch.dtype, max_new: int = 8) -> dict:
    prompt = fixture["provenance"]["prompt"]
    ids = [1, *encode(prompt), 7]
    generated = []
    ended = False
    for _ in range(max_new):
        output = model(torch.tensor([ids]), fixture["images"].to(dtype), fixture["geometry"])
        logits = output["logits"][0, -1].float()
        logits[:2] = -torch.inf
        logits[3:10] = -torch.inf
        logits[266:] = -torch.inf
        token = int(logits.argmax())
        if token == 2:
            ended = True
            break
        generated.append(token)
        ids.append(token)
    try:
        text, valid = decode(generated), True
    except UnicodeDecodeError:
        text, valid = bytes(t - 10 for t in generated).decode("utf-8", errors="replace"), False
    return {"text": text, "eos": ended, "valid_utf8": valid}


def resource_gate(cfg: ScaleConfig, dtype: torch.dtype, output: Path,
                  allow_large: bool, mapped: bool, streaming: bool) -> dict:
    report = parameter_report(build(cfg, device="meta"))
    count = report["unique_parameters"]
    if count > 1_100_000_000:
        raise ValueError("Bounded scale pilot supports at most 1.1B parameters")
    if count >= 100_000_000 and not allow_large:
        raise ValueError("Large allocation requires explicit --allow-large (count is safe without it)")
    weights = count * torch.empty((), dtype=dtype).element_size()
    # Mapped weights are file-backed/reclaimable; this is a guard, not an OS memory quota.
    required_ram = (512 * 1024**2 if mapped else weights) + 512 * 1024**2
    if not streaming:
        required_ram += weights
    try:
        limit = (Path("/sys/fs/cgroup/memory.max").read_text()).strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        available = int(limit) - current if limit != "max" else None
    except (OSError, ValueError):
        available = None
    if available is not None and available < required_ram:
        raise MemoryError("Insufficient memory estimate; use mapped/streaming mode or a larger runner")
    parent = output
    while not parent.exists():
        parent = parent.parent
    required_disk = weights * (3 if mapped else 2) + 128 * 1024**2
    if shutil.disk_usage(parent).free < required_disk:
        raise OSError("Insufficient free storage for backing plus resumable checkpoints")
    return {"estimated_ram_bytes": required_ram, "cgroup_available_bytes": available,
            "estimated_disk_bytes": required_disk, "file_backed_parameters": mapped}


def run(cfg: ScaleConfig, output: Path, *, steps: int = 8, dtype_name: str = "float32",
        lr: float = 0.003, threads: int = 2, allow_large: bool = False, mapped: bool = False,
        streaming: bool = False, resume: bool = False, seconds: int = 600) -> dict:
    if (type(steps) is not int or not 1 <= steps <= 64 or type(threads) is not int
            or not 1 <= threads <= 4 or type(seconds) is not int or not 1 <= seconds <= 1200
            or type(lr) not in (float, int) or not math.isfinite(lr) or not 0 < lr <= 0.01):
        raise ValueError("Invalid bounded training options")
    if dtype_name not in ("float32", "bfloat16") or cfg.vocab_size != 266:
        raise ValueError("Scale pilot requires FP32/BF16 and the declared byte vocabulary")
    dtype = getattr(torch, dtype_name)
    resources = resource_gate(cfg, dtype, output, allow_large, mapped, streaming)
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=resume)
    lock = output / "RUNNING.lock"
    with lock.open("x") as stream:
        stream.write("Single scale training writer; stale lock requires inspecting prior failure.")
    optimizer = None
    started = time.monotonic()
    starting_step = 0
    backing = None
    try:
        if resume:
            previous_folder = get_latest(output)
            old = index_at(previous_folder)
            dtype_old = next(iter(old["shards"][0]["tensors"].values()))["dtype"]
            if (old["config"] != asdict(cfg) or old["source_sha256"] != source_hash()
                    or old["optimizer"]["lr"] != lr or dtype_old != str(dtype)
                    or old["torch"] != str(torch.__version__)):
                raise ValueError("Resume config, dtype, source, optimizer or library mismatch")
            starting_step = old["training"]["completed_steps"]
        if starting_step + steps > 64:
            raise ValueError("Cumulative scale pilot limit is 64 steps; no automatic infinite training")
        if mapped:
            backing = output / f"parameters-{starting_step:04d}.bin"
        torch.manual_seed(20260929)
        model = build(cfg, dtype=dtype, initialize=not resume, backing_file=backing)
        if resume:
            old = load_into(model, previous_folder)
            torch.set_rng_state(torch.tensor(old["torch_rng"], dtype=torch.uint8))
        counts = parameter_report(model)
        print(json.dumps({"phase": "real_parameters_allocated", **counts}), flush=True)
        optimizer = AuditedSGD(model, lr, streaming)
        fixture = example(0, cfg.max_pages, 128)
        input_data = {k: v for k, v in fixture.items() if k != "provenance"}
        input_data["images"] = input_data["images"].to(dtype)
        with torch.no_grad():
            initial_loss = float(model(**input_data)["loss"])
        audits = []
        observed_fixture = []
        for step in range(starting_step, starting_step + steps):
            if time.monotonic() - started > seconds:
                break
            sample = example(step % 8, cfg.max_pages, 128)
            observed_fixture.append(sample["provenance"])
            data = {k: v for k, v in sample.items() if k != "provenance"}
            data["images"] = data["images"].to(dtype)
            out = model(**data)
            value = float(out["loss"].detach())
            audit = optimizer.backward(out["loss"])
            record = {"step": step + 1, "loss": value, **audit}
            audits.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "by_tensor"}), flush=True)
            del out
        completed = starting_step + len(audits)
        if not audits:
            raise TimeoutError("No scale training step completed within budget")
        optimizer.close()
        optimizer = None
        with torch.no_grad():
            final_loss = float(model(**input_data)["loss"])
        prediction = predict(model, fixture, dtype)
        training = {"completed_steps": completed, "learning_rate": lr,
                    "starting_step_this_run": starting_step, "steps_this_run": len(audits),
                    "dtype": dtype_name, "streaming_sgd": streaming,
                    "corpus": "tiny_programmatic_engineering_fixture", "full_pretraining": False,
                    "gradient_accumulation": False, "momentum": 0, "global_clipping": False}
        checkpoint_folder = output / f"checkpoint-{completed:04d}"
        index = save(model, checkpoint_folder, training)
        set_latest(output, checkpoint_folder)
        report = {"status": "scale_pilot_complete" if len(audits) == steps else "time_bounded_partial_pilot",
                  "parameters": counts, "training": training, "requested_steps_this_run": steps,
                  "initial_fixture_loss": initial_loss, "final_fixture_loss": final_loss,
                  "audit": audits, "data_provenance": observed_fixture,
                  "prediction_fixture0": prediction, "reference_fixture0": fixture["provenance"]["target"],
                  "prediction_is_quality_benchmark": False,
                  "checkpoint_shards": len(index["shards"]),
                  "checkpoint_total_file_bytes": sum(s["file_bytes"] for s in index["shards"]),
                  "resources": resources, "elapsed_seconds": time.monotonic() - started,
                  "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "device": "cpu", "threads": threads},
                  "source_sha256": source_hash(), "quality_certified": False}
        write_json(output / f"report-{completed:04d}.json", report)
        print(json.dumps({"phase": "checkpoint_saved", "step": completed,
                          "shards": report["checkpoint_shards"], "final_loss": final_loss}), flush=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        failure = output / f"FAILED-{time.time_ns()}.json"
        write_json(failure, {"exception": type(error).__name__, "message": str(error),
                             "discard_in_memory_state": True, "resume_from_last_committed_checkpoint_only": True})
        raise
    finally:
        if optimizer is not None:
            optimizer.close()
        lock.unlink(missing_ok=True)
        # Mapping file is an expendable work area, never the authoritative checkpoint.
        # Keep it on Windows if still mapped; no silent claim that cleanup succeeded there.
        if backing is not None and platform.system() != "Windows":
            backing.unlink(missing_ok=True)


def verify(run_folder: Path, allow_large: bool, mapped: bool = False) -> dict:
    checkpoint_folder = get_latest(run_folder)
    index = index_at(checkpoint_folder)
    cfg = ScaleConfig(**index["config"])
    dtype = getattr(torch, index["training"]["dtype"])
    torch.set_num_threads(2)
    resource_gate(cfg, dtype, run_folder, allow_large, mapped, True)
    backing = run_folder / "verification-parameters.bin" if mapped else None
    model = build(cfg, dtype, initialize=False, backing_file=backing)
    try:
        load_into(model, checkpoint_folder)
        step = index["training"]["completed_steps"]
        report = json.loads((run_folder / f"report-{step:04d}.json").read_text(encoding="utf-8"))
        fixture = example(0, cfg.max_pages, 128)
        prediction = predict(model, fixture, dtype)
        result = {"parameters_reloaded": parameter_report(model)["unique_parameters"],
                  "matches_saved_prediction": prediction == report["prediction_fixture0"],
                  "source_sha256": source_hash(), "checkpoint_step": step,
                  "prediction": prediction, "quality_certified": False}
        if not result["matches_saved_prediction"]:
            raise ValueError("Independent checkpoint reload changed the fixture prediction")
        write_json(run_folder / f"reload-{step:04d}.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        del model
        gc.collect()
        if backing is not None and platform.system() != "Windows":
            backing.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="H2LM real 1B scale prototype, NOT pretrained intelligence")
    parser.add_argument("command", choices=("count", "train", "verify"))
    parser.add_argument("--config", type=Path, default=Path("configs/model/h2lm_v1_1b.yaml"))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--mapped", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = tiny_config() if args.tiny else ScaleConfig.read(args.config)
    if args.command == "count":
        print(json.dumps(parameter_report(build(cfg, device="meta")), indent=2))
    elif args.output is None:
        parser.error("--output is required for train/verify")
    elif args.command == "verify":
        verify(args.output, args.allow_large, args.mapped)
    else:
        run(cfg, args.output, steps=args.steps, dtype_name=args.dtype, threads=args.threads,
            allow_large=args.allow_large, mapped=args.mapped, streaming=args.streaming, resume=args.resume)


if __name__ == "__main__":
    main()
