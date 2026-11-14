from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from src.eval.audio_io import load_mono, missing_files
from src.eval.evaluate import EvalError
from src.eval.pool import Pool, load_pool

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "laion/clap-htsat-unfused"

## takes a list of mono waveforms, returns an (n, d) array of embeddings
EmbedBatch = Callable[[list[np.ndarray]], np.ndarray]


def make_clap_embedder(model_name: str) -> tuple[EmbedBatch, int]:
    ## torch and transformers are imported here so the tests run without them
    import torch
    from transformers import AutoFeatureExtractor, ClapModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(model_name).to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    sample_rate = int(extractor.sampling_rate)

    def embed_batch(waves: list[np.ndarray]) -> np.ndarray:
        inputs = extractor(waves, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            out = model.get_audio_features(**inputs)
        ## some transformers versions return an output object instead of a tensor
        features = out if isinstance(out, torch.Tensor) else out.pooler_output
        return features.cpu().numpy()

    return embed_batch, sample_rate


### embed a list of audio paths in batches, normalizing the embeddings
def embed_paths(
    paths: Sequence[str],
    embed_batch: EmbedBatch,
    sample_rate: int,
    batch_size: int = 16,
    label: str = "",
) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), batch_size):
        waves = [load_mono(p, sample_rate) for p in paths[start : start + batch_size]]
        chunk = np.asarray(embed_batch(waves), dtype=np.float64)
        if chunk.ndim != 2 or len(chunk) != len(waves):
            raise EvalError(f"embedder returned shape {chunk.shape} for {len(waves)} clips")
        chunks.append(chunk)
        done = min(start + batch_size, len(paths))
        if (start // batch_size) % 10 == 0 or done == len(paths):
            print(f"  {label} {done}/{len(paths)}", flush=True)
    emb = np.concatenate(chunks)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if not (norms > 0).all() or not np.isfinite(emb).all():
        raise EvalError("some embeddings are zero or not finite")
    return (emb / norms).astype(np.float32)


## build a system file with query and library embeddings, and save it to out_path
def build_system_file(
    pool: Pool,
    embed_batch: EmbedBatch,
    sample_rate: int,
    out_path: Path,
    batch_size: int = 16,
) -> Path:
    paths = pool.library["audio_path"].tolist() + pool.queries["audio_path"].tolist()
    missing = missing_files(paths)
    if missing:
        raise EvalError(
            f"{len(missing)} of {len(paths)} audio files are missing, e.g. {missing[:3]}"
        )
    library_emb = embed_paths(
        pool.library["audio_path"].tolist(), embed_batch, sample_rate, batch_size, "library"
    )
    query_emb = embed_paths(
        pool.queries["audio_path"].tolist(), embed_batch, sample_rate, batch_size, "queries"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        query_ids=pool.queries["query_id"].to_numpy(dtype=str),
        library_ids=pool.library["clip_id"].to_numpy(dtype=str),
        query_emb=query_emb,
        library_emb=library_emb,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed the test pool with CLAP (baseline B1).")
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv")
    parser.add_argument("--relevance", type=Path, default=REPO_ROOT / "manifests/relevance.json")
    parser.add_argument("--pool-kits", nargs="+", metavar="KIT_ID")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results/embeddings/clap_syn_beatbox.npz"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--selftest", action="store_true", help="run the model on noise and exit")
    args = parser.parse_args()
    try:
        if args.selftest:
            embed_batch, sample_rate = make_clap_embedder(args.model)
            rng = np.random.default_rng(0)
            noise = [rng.normal(0, 0.1, 2 * sample_rate).astype(np.float32) for _ in range(2)]
            result = embed_batch(noise)
            shape = result.shape
            print(f"model ok: {shape[0]} embeddings, dimension {shape[1]}, rate {sample_rate}")
            return
        pool = load_pool(args.library, args.queries, args.relevance, pool_kits=args.pool_kits)
        print(f"{len(pool.library)} library clips, {len(pool.queries)} queries")
        ## check the audio before loading the model, which is slow and may download
        paths = pool.library["audio_path"].tolist() + pool.queries["audio_path"].tolist()
        missing = missing_files(paths)
        if missing:
            raise EvalError(
                f"{len(missing)} of {len(paths)} audio files are missing, e.g. {missing[:3]}"
            )
        embed_batch, sample_rate = make_clap_embedder(args.model)
        path = build_system_file(pool, embed_batch, sample_rate, args.out, args.batch_size)
    except ImportError as error:
        parser.exit(
            1,
            f"Missing library ({error}). Install: pip install torch transformers soundfile scipy\n",
        )
    except (OSError, EvalError) as error:
        parser.exit(1, f"CLAP baseline failed: {error}\n")
    print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
