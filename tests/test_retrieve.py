from src.rag.retrieve import reciprocal_rank_fusion


def test_rrf_rewards_agreement_between_rankers():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[:2] == [1, 3]  # ranked well by both lists
    assert set(ranked) == {1, 2, 3, 4}


def test_rrf_single_list_keeps_order():
    scores = reciprocal_rank_fusion([[5, 6, 7]])
    assert sorted(scores, key=scores.get, reverse=True) == [5, 6, 7]


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == {}
