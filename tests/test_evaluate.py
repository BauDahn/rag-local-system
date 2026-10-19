from src.rag.evaluate import contains_keyword, first_relevant_rank


def test_keyword_matches_whole_words_only():
    assert contains_keyword("He drew a boa constrictor.", ["boa"])
    assert not contains_keyword("He rowed the boat.", ["boa"])
    assert not contains_keyword("The fox was tamed.", ["tame"])


def test_keyword_is_case_insensitive_and_handles_symbols():
    assert contains_keyword("Asteroid B-612 is tiny", ["b-612"])


def test_first_relevant_rank_is_one_based():
    texts = ["nothing", "a sheep here", "sheep again"]
    assert first_relevant_rank(texts, ["sheep"]) == 2
    assert first_relevant_rank(texts, ["fox"]) is None
