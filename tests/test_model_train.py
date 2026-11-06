"""End to end on tiny made-up audio: manifests -> features -> training -> .npz -> D's score."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # part of the optional "model" extra

import soundfile as sf  # noqa: E402

from src.eval.evaluate import evaluate  # noqa: E402
from src.models.config import load_model_config  # noqa: E402
from src.models.data import LIBRARY_COLUMNS, QUERY_COLUMNS, FeatureCache, load_clips  # noqa: E402
from src.models.embed import embed_clips, load_run  # noqa: E402
from src.models.export import write_embeddings  # noqa: E402
from src.models.train import train  # noqa: E402
from src.music.spec import load_spec  # noqa: E402

SR = 22050


def hits_audio(times, freq, seconds=2.0):
    audio = np.zeros(int(seconds * SR), np.float32)
    t = np.arange(int(0.08 * SR)) / SR
    hit = (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.02)).astype(np.float32)
    for s in times:
        i = int(s * SR)
        audio[i : i + hit.size] += hit[: audio.size - i]
    return audio


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """12 rhythms (6 train, 3 validation, 3 test): a low 'kit' loop and two high 'queries'."""
    root = tmp_path_factory.mktemp("e2e")
    rng = np.random.default_rng(0)
    lib, qry = [], []
    for p in range(12):
        split = "train" if p < 6 else "validation" if p < 9 else "test"
        times = np.sort(rng.uniform(0.05, 1.8, size=6))
        path = root / f"lib{p}.wav"
        sf.write(path, hits_audio(times, 60), SR)
        lib.append(
            dict.fromkeys(LIBRARY_COLUMNS, "x")
            | {
                "clip_id": f"L{p}",
                "pattern_id": f"P{p}",
                "dualized_pattern_id": f"D{p}",
                "split": split,
                "audio_path": str(path),
            }
        )
        for k in range(2):
            path = root / f"q{p}_{k}.wav"
            sf.write(path, hits_audio(times + rng.uniform(-0.01, 0.01, 6), 3000), SR)
            qry.append(
                dict.fromkeys(QUERY_COLUMNS, "x")
                | {
                    "query_id": f"Q{p}_{k}",
                    "pattern_id": f"P{p}",
                    "dualized_pattern_id": f"D{p}",
                    "query_type": "beatbox",
                    "split": split,
                    "audio_path": str(path),
                }
            )
    pd.DataFrame(lib).to_csv(root / "library.csv", index=False)
    pd.DataFrame(qry).to_csv(root / "queries.csv", index=False)
    return root


def test_train_embed_and_score(dataset, tmp_path):
    cfg = load_model_config()
    cfg = dataclasses.replace(
        cfg, training=dataclasses.replace(cfg.training, epochs=2, batch_patterns=3)
    )
    clips = load_clips(dataset / "library.csv", dataset / "queries.csv")
    cache = FeatureCache.build(clips, "r1", cfg, tmp_path / "r1.npz", log=lambda *_: None)
    run = tmp_path / "run"
    summary = train(
        cfg=cfg,
        spec=load_spec(),
        clips=clips,
        cache=cache,
        front_end="r1",
        out_dir=run,
        device=torch.device("cpu"),
        log=lambda *_: None,
    )

    for name in ("best.pt", "last.pt", "log.csv", "run.json"):
        assert (run / name).exists()
    log = pd.read_csv(run / "log.csv")
    assert list(log.epoch) == [1, 2]
    assert {"train_loss", "val_mrr", "val_hit@1", "val_chance_mrr"} <= set(log.columns)
    assert json.loads((run / "run.json").read_text())["train_patterns"] == 6
    assert summary["best_epoch"] in (1, 2)

    # Reload the best checkpoint, embed the test split, and score it with D's code.
    model, cfg2, front_end, ckpt = load_run(run, "best", torch.device("cpu"))
    assert front_end == "r1" and cfg2.encoder == cfg.encoder
    test = clips[clips.split == "test"]
    library, queries = test[test.kind == "library"], test[test.kind != "library"]
    q_emb = embed_clips(model, queries, cache, torch.device("cpu"))
    l_emb = embed_clips(model, library, cache, torch.device("cpu"))
    path = write_embeddings(tmp_path / "s_t.npz", list(queries.id), q_emb, list(library.id), l_emb)
    relevance = {q: [f"L{q[1:].split('_')[0]}"] for q in queries.id}
    table, counts = evaluate(path, relevance)
    assert counts["n_queries_scored"] == 6
    assert table.rr.between(1 / 3, 1).all()


def test_reloaded_model_matches_the_trained_one(dataset, tmp_path):
    cfg = load_model_config()
    cfg = dataclasses.replace(
        cfg, training=dataclasses.replace(cfg.training, epochs=1, batch_patterns=3)
    )
    clips = load_clips(dataset / "library.csv", dataset / "queries.csv")
    cache = FeatureCache.build(clips, "r3", cfg, tmp_path / "r3.npz", log=lambda *_: None)
    train(
        cfg=cfg,
        spec=load_spec(),
        clips=clips,
        cache=cache,
        front_end="r3",
        out_dir=tmp_path / "run",
        device=torch.device("cpu"),
        log=lambda *_: None,
    )
    a, *_ = load_run(tmp_path / "run", "last")
    b, *_ = load_run(tmp_path / "run", "last")
    some = clips.head(4)
    np.testing.assert_allclose(
        embed_clips(a, some, cache, torch.device("cpu")),
        embed_clips(b, some, cache, torch.device("cpu")),
        atol=1e-6,
    )
