from french_mining.frequency import FrequencyList, load_exclusion_list


def test_loads_default_frequency_list():
    freq = FrequencyList.load()
    assert freq.rank("le") == 1
    assert freq.is_within_floor("être", floor=500)
    assert not freq.is_within_floor("chèvrefeuille", floor=500)


def test_rank_is_case_and_whitespace_insensitive():
    freq = FrequencyList.load()
    assert freq.rank(" Être ") == freq.rank("être")


def test_missing_exclusion_file_returns_empty_set():
    assert load_exclusion_list("/nonexistent/path.txt") == frozenset()


def test_exclusion_list_parses_lines_and_comments(tmp_path):
    p = tmp_path / "exclusions.txt"
    p.write_text("chat\n# a comment\n\nCHIEN\n")
    excluded = load_exclusion_list(str(p))
    assert excluded == frozenset({"chat", "chien"})
