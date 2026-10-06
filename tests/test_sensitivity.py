from dema.experiments.sensitivity import replay_alpha, replay_tau


def _document():
    return {
        "predictions": {
            "source": [
                {
                    "target_column": "a", "retrieval_rank": 1,
                    "jev_score": 0.2, "coma_plus_score": 0.5,
                    "fusion_score": 0.38, "jina_rank": 2,
                },
                {
                    "target_column": "b", "retrieval_rank": 2,
                    "jev_score": 0.5, "coma_plus_score": 0.0,
                    "fusion_score": 0.2, "jina_rank": 1,
                },
                {"target_column": "tail", "retrieval_rank": 3},
            ]
        }
    }


def test_alpha_replay_uses_raw_component_scores_without_jina():
    ranking = replay_alpha(_document(), 1.0)
    assert ranking["source"] == [("b", 0.5), ("a", 0.2), ("tail", 0.0)]


def test_tau_replay_matches_gate_boundary_and_saved_jina_order():
    bypassed, low = replay_tau(_document(), 0.18)
    refined, high = replay_tau(_document(), 0.181)
    assert bypassed["source"][0][0] == "a"  # strict margin < tau
    assert low["activations"] == 0
    assert refined["source"][0][0] == "b"
    assert high == {"evaluations": 1, "disagreements": 1, "activations": 1}
