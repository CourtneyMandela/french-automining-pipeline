from unittest.mock import MagicMock

from french_mining.queue_ordering import (
    get_new_backlog_note_ids,
    merge_priority_with_backlog,
    reorder_queue,
)


def test_get_new_backlog_note_ids_queries_and_sorts_by_due():
    client = MagicMock()
    client.find_cards.return_value = [101, 102, 103]
    client.cards_info.return_value = [
        {"cardId": 101, "note": 1, "due": 5},
        {"cardId": 102, "note": 2, "due": 1},
        {"cardId": 103, "note": 3, "due": 3},
    ]

    result = get_new_backlog_note_ids(client, "French Sentence Mining", "French::Mining")

    client.find_cards.assert_called_once_with(
        'deck:"French::Mining" note:"French Sentence Mining" is:new'
    )
    assert result == [2, 3, 1]  # ordered by ascending due


def test_merge_priority_with_backlog_dedups_and_preserves_backlog_order():
    merged = merge_priority_with_backlog(
        prioritized_note_ids=[10, 20],
        backlog_note_ids=[20, 30, 10, 40],
    )
    assert merged == [10, 20, 30, 40]


def test_merge_priority_with_backlog_handles_empty_backlog():
    merged = merge_priority_with_backlog(prioritized_note_ids=[1, 2], backlog_note_ids=[])
    assert merged == [1, 2]


def test_reorder_queue_assigns_sequential_due_in_ranked_order():
    client = MagicMock()
    client.notes_info.return_value = [
        {"noteId": 1, "cards": [101]},
        {"noteId": 2, "cards": [102]},
        {"noteId": 3, "cards": [103]},
    ]
    client.cards_info.return_value = [
        {"cardId": 101, "type": 0},
        {"cardId": 102, "type": 0},
        {"cardId": 103, "type": 0},
    ]

    rewritten = reorder_queue(client, ranked_note_ids=[3, 1, 2])

    assert rewritten == [103, 101, 102]
    calls = client.set_specific_value_of_card.call_args_list
    assert calls[0].args == (103, "due", "0")
    assert calls[1].args == (101, "due", "1")
    assert calls[2].args == (102, "due", "2")


def test_reorder_queue_skips_notes_whose_card_already_left_new_state():
    client = MagicMock()
    client.notes_info.return_value = [
        {"noteId": 1, "cards": [101]},
        {"noteId": 2, "cards": [102]},  # already in review
    ]
    client.cards_info.return_value = [
        {"cardId": 101, "type": 0},
        {"cardId": 102, "type": 2},  # review card -- FSRS owns this now
    ]

    rewritten = reorder_queue(client, ranked_note_ids=[2, 1])

    assert rewritten == [101]
    client.set_specific_value_of_card.assert_called_once_with(101, "due", "0")


def test_reorder_queue_respects_start_due_offset():
    client = MagicMock()
    client.notes_info.return_value = [{"noteId": 1, "cards": [101]}]
    client.cards_info.return_value = [{"cardId": 101, "type": 0}]

    reorder_queue(client, ranked_note_ids=[1], start_due=50)

    client.set_specific_value_of_card.assert_called_once_with(101, "due", "50")


def test_reorder_queue_handles_empty_input_without_calling_client():
    client = MagicMock()
    rewritten = reorder_queue(client, ranked_note_ids=[])
    assert rewritten == []
    client.notes_info.assert_not_called()
