"""A causal claim: an explicit, versionable set of assumptions (a DAG over named variables).

The agent (Claude) writes claims as YAML; everything downstream is deterministic:
which variables to adjust for, which are forbidden, and which conditional
independencies the claim implies (= what the data can falsify).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import networkx as nx
import yaml


@dataclass
class Edge:
    src: str
    dst: str
    confidence: str = "medium"   # high | medium | low
    why: str = ""


@dataclass
class Claim:
    treatment: str
    outcome: str
    observed: list[str]                       # columns present in the data
    edges: list[Edge]
    latent: list[str] = field(default_factory=list)   # hypothesised unmeasured variables
    question: str = ""
    author: str = ""                          # e.g. "claude (blind)", "oracle", "wrong:mediator-as-confounder"
    notes: str = ""
    # Pre-treatment covariates whose mutual relations we make NO claim about: expanded to a
    # complete DAG among themselves (in listed order), so no independencies are asserted between them.
    background: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Claim":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw["edges"] = [Edge(**e) if isinstance(e, dict) else Edge(*e) for e in raw["edges"]]
        return cls(**raw)

    def dump(self, path: str | Path) -> None:
        d = {"question": self.question, "author": self.author, "treatment": self.treatment,
             "outcome": self.outcome, "observed": self.observed, "latent": self.latent,
             "background": self.background, "edges": [e.__dict__ for e in self.edges], "notes": self.notes}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100),
                              encoding="utf-8")

    # ------------------------------------------------------------------ graph
    @property
    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.observed + self.latent)
        g.add_edges_from((e.src, e.dst) for e in self.edges)
        g.add_edges_from((a, b) for i, a in enumerate(self.background) for b in self.background[i + 1:])
        unknown = set(g.nodes) - set(self.observed) - set(self.latent)
        if unknown:
            raise ValueError(f"edges mention variables that are neither observed nor latent: {unknown}")
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError(f"claim is not acyclic: {nx.find_cycle(g)}")
        return g

    def descendants_of_treatment(self) -> set[str]:
        return nx.descendants(self.graph, self.treatment)

    def is_valid_adjustment(self, S: set[str]) -> bool:
        """Backdoor criterion (under THIS claim): no descendants of T, and S blocks every
        backdoor path T <- ... -> Y."""
        g, T, Y = self.graph, self.treatment, self.outcome
        if S & nx.descendants(g, T):
            return False
        gb = g.copy()
        gb.remove_edges_from(list(g.out_edges(T)))
        return nx.is_d_separator(gb, {T}, {Y}, set(S))

    def adjustment_set(self) -> set[str] | None:
        """A minimal valid backdoor set using only observed, non-descendant variables;
        None if the claim implies the effect is NOT identifiable by adjustment."""
        g, T, Y = self.graph, self.treatment, self.outcome
        allowed = set(self.observed) - nx.descendants(g, T) - {T, Y}
        gb = g.copy()
        gb.remove_edges_from(list(g.out_edges(T)))
        if nx.is_d_separator(gb, {T}, {Y}, set()):
            return set()
        sep = nx.find_minimal_d_separator(gb, {T}, {Y}, restricted=allowed | {T, Y})
        return None if sep is None else set(sep)

    def forbidden_controls(self) -> dict[str, str]:
        """Observed variables an analyst might be tempted to adjust for, with the reason not to."""
        g, T, Y = self.graph, self.treatment, self.outcome
        out = {}
        for v in set(self.observed) - {T, Y}:
            if v in nx.descendants(g, T):
                out[v] = ("mediator (on a causal path from treatment to outcome)"
                          if v in nx.ancestors(g, Y) else "post-treatment variable (descendant of treatment)")
            elif not self.is_valid_adjustment((self.adjustment_set() or set()) | {v}) and \
                    self.adjustment_set() is not None:
                out[v] = "opens a non-causal path when adjusted for (collider / M-bias)"
        return out

    def implied_independencies(self) -> list[tuple[str, str, tuple[str, ...]]]:
        """Testable consequences: for every non-adjacent pair of OBSERVED variables that the
        claim says can be d-separated using observed variables only, (A, B, S) meaning A ⟂ B | S.
        Returned with B later than A in a topological order (B is used as the CRT target)."""
        g = self.graph
        order = {v: i for i, v in enumerate(nx.topological_sort(g))}
        obs = set(self.observed)
        out = []
        for a, b in combinations(sorted(obs, key=order.get), 2):
            if g.has_edge(a, b) or g.has_edge(b, a):
                continue
            sep = nx.find_minimal_d_separator(g, {a}, {b}, restricted=obs)
            if sep is not None:
                out.append((a, b, tuple(sorted(sep, key=order.get))))
        return out

    def summary(self) -> dict:
        adj = self.adjustment_set()
        return {"identifiable_by_adjustment": adj is not None,
                "adjustment_set": sorted(adj) if adj is not None else None,
                "forbidden": self.forbidden_controls(),
                "implied_independencies": [f"{a} ⟂ {b} given {{{', '.join(s)}}}"
                                           for a, b, s in self.implied_independencies()]}
