from src.data.splits import load_splits


def test_no_overlap():
    splits = load_splits()
    train = set(splits["train"])
    val = set(splits["validation"])
    test = set(splits["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_union_covers_all():
    splits = load_splits()
    all_ids = set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
    assert len(all_ids) == 700


def test_split_sizes():
    splits = load_splits()
    assert len(splits["train"]) == 500
    assert len(splits["validation"]) == 100
    assert len(splits["test"]) == 100
