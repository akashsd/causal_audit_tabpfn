"""CausalDS (arXiv 2607.08093, Apache-2.0 code / CC0 data) as a ground-truth benchmark.

Layout (blinding: the agent only ever gets the `public` folder):
  data/raw/causalds/grading.parquet              catalog incl. private ground truth
  data/raw/causalds/public/<scene>/              story.md, columns.json, data.parquet
  data/raw/causalds/private/<scene>.json         ground truth (graph, true ATE, valid sets)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from .backend import DATA

from .claim import Claim, Edge

ROOT = DATA / "raw" / "causalds"
HF = "https://huggingface.co/datasets/andleb/causalds/resolve/main/"


def catalog() -> pd.DataFrame:
    path = ROOT / "grading.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF + "catalogs/grading.parquet", path)
    return pd.read_parquet(path).drop_duplicates("scene_id").set_index("scene_id")


def ground_truth(scene: str) -> dict:
    p = ROOT / "private" / f"{scene}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(catalog().loc[scene, "ground_truth_json"], encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def fetch(scene: str, variant: str = "clean") -> Path:
    """Download the public files for a scene; returns the public folder."""
    d = ROOT / "public" / scene
    if not (d / "data.parquet").exists():
        d.mkdir(parents=True, exist_ok=True)
        base = f"{HF}benchmark/main/scenes/{scene}/"
        urllib.request.urlretrieve(base + "story.md", d / "story.md")
        urllib.request.urlretrieve(base + f"variants/{variant}/data.parquet", d / "data.parquet")
        gt = ground_truth(scene)["graph"]
        cols = list(pd.read_parquet(d / "data.parquet").columns)
        (d / "columns.json").write_text(json.dumps({
            "columns": cols, "treatment": gt["treatment_named"], "outcome": gt["outcome_named"]},
            indent=1, ensure_ascii=False), encoding="utf-8")
    return d


def load_data(scene: str) -> pd.DataFrame:
    return pd.read_parquet(fetch(scene) / "data.parquet")


def oracle_claim(scene: str) -> Claim:
    gt = ground_truth(scene)
    g, m = gt["graph"], gt["mapping"]
    return Claim(treatment=g["treatment_named"], outcome=g["outcome_named"],
                 observed=g["observed_nodes_named"], latent=g["latent_nodes_named"],
                 edges=[Edge(m[a], m[b], "high", "ground truth") for a, b in g["edges"]],
                 author="oracle (CausalDS ground truth)")


def true_ate(scene: str) -> float:
    return float(ground_truth(scene)["causal"]["true_ate"]["value"])


def scene_table() -> pd.DataFrame:
    rows = []
    for sid, r in catalog().iterrows():
        gt = json.loads(r.ground_truth_json)
        G, c, md = gt["graph"], gt["causal"], gt["metadata"]
        rows.append({"scene": sid, "structure": r.scene_structure_label, "variant": r.observation_variant,
                     "treatment_type": md["treatment_contrast"]["treatment_type"],
                     "outcome_type": r.outcome_type, "n_observed": len(G["observed_nodes"]),
                     "n_latent": len(G["latent_nodes"]), "identifiable": c["identification"]["identifiable"],
                     "method": c["identification"]["method"], "true_ate": c["true_ate"]["value"],
                     "domain": md.get("proposed_domain", "")})
    return pd.DataFrame(rows)


def benchmark_scenes() -> dict[str, list[str]]:
    """Clean, binary-treatment scenes: identifiable ones (estimation) and hidden-confounder
    ones (should be flagged, not estimated)."""
    t = scene_table()
    t = t[(t.variant == "clean") & (t.treatment_type == "binary")]
    return {"identifiable": sorted(t[t.method.isin(["backdoor", "trivial_zero"])].scene),
            "hidden_confounding": sorted(t[(t.method == "none") & (t.n_latent > 0)].scene)}


def replicates(df: pd.DataFrame, n: int, max_reps: int, seed: int = 0) -> list[pd.DataFrame]:
    """Disjoint subsamples = independent draws from the same SCM (for coverage)."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [shuffled.iloc[i * n:(i + 1) * n].reset_index(drop=True)
            for i in range(min(max_reps, len(df) // n))]
