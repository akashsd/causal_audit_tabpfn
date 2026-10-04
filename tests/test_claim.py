from causalaudit.claim import Claim, Edge


def claim(edges, observed, latent=(), T="T", Y="Y"):
    return Claim(T, Y, list(observed), [Edge(a, b) for a, b in edges], list(latent))


def test_confounder_is_adjusted():
    c = claim([("Z", "T"), ("Z", "Y"), ("T", "Y")], ["Z", "T", "Y"])
    assert c.adjustment_set() == {"Z"}
    assert c.forbidden_controls() == {}


def test_mediator_is_forbidden():
    c = claim([("T", "M"), ("M", "Y")], ["T", "M", "Y"])
    assert c.adjustment_set() == set()
    assert "mediator" in c.forbidden_controls()["M"]
    assert not c.is_valid_adjustment({"M"})


def test_collider_is_forbidden():
    c = claim([("T", "C"), ("Y", "C"), ("T", "Y")], ["T", "C", "Y"])
    assert c.adjustment_set() == set()
    assert "C" in c.forbidden_controls()


def test_m_bias_pretreatment_collider():
    # T <- A -> M <- B -> Y : M is pre-treatment but adjusting for it opens a path
    c = claim([("A", "T"), ("A", "M"), ("B", "M"), ("B", "Y"), ("T", "Y")], ["A", "B", "M", "T", "Y"])
    assert c.is_valid_adjustment(set())
    assert not c.is_valid_adjustment({"M"})
    assert "M" in c.forbidden_controls()


def test_hidden_confounder_not_identifiable_and_iv_untestable():
    c = claim([("Z", "T"), ("T", "Y"), ("U", "T"), ("U", "Y")], ["Z", "T", "Y"], latent=["U"])
    assert c.adjustment_set() is None
    assert c.implied_independencies() == []          # true IV graph implies nothing testable
    naive = claim([("Z", "T"), ("T", "Y")], ["Z", "T", "Y"])
    assert naive.implied_independencies() == [("Z", "Y", ("T",))]  # ...but "no confounding" does


def test_background_makes_no_independence_claims():
    c = Claim("T", "Y", ["A", "B", "T", "Y"], [Edge("A", "T"), Edge("B", "Y"), Edge("T", "Y")],
              background=["A", "B"])
    assert ("A", "B", ()) not in c.implied_independencies()
    assert c.adjustment_set() in ({"A"}, {"B"}, {"A", "B"})


def test_roundtrip(tmp_path):
    c = claim([("Z", "T"), ("Z", "Y"), ("T", "Y")], ["Z", "T", "Y"])
    c.dump(tmp_path / "c.yaml")
    assert Claim.load(tmp_path / "c.yaml").adjustment_set() == {"Z"}


def test_markov_equivalent_wrong_claim_is_undetectable():
    from causalaudit.bench import detectable
    truth = claim([("Z", "T"), ("Z", "Y"), ("T", "Y")], ["Z", "T", "Y"])
    flipped = claim([("T", "Z"), ("Z", "Y"), ("T", "Y")], ["Z", "T", "Y"])  # complete triangle: same (no) implications
    assert flipped.adjustment_set() != truth.adjustment_set()
    assert not detectable(flipped, truth)
    iv_truth = claim([("Z", "T"), ("T", "Y"), ("U", "T"), ("U", "Y")], ["Z", "T", "Y"], latent=["U"])
    no_confounding = claim([("Z", "T"), ("T", "Y")], ["Z", "T", "Y"])
    assert detectable(no_confounding, iv_truth)
