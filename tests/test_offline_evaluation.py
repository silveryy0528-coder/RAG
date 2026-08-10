import json
from pathlib import Path

import rag.offline_evaluation as offline_evaluation


def test_normalize_text_WHEN_text_contains_punctuation_THEN_returns_lowercase_words():
    assert offline_evaluation.normalize_text("Hello, World!") == "hello world"


def test_exact_match_and_f1_WHEN_prediction_is_similar_THEN_scores_are_correct():
    prediction = "Automatic parameter selection"
    reference = "automatic parameter selection"

    assert offline_evaluation.exact_match(prediction, reference)
    assert offline_evaluation.f1_score(prediction, reference) == 1.0

    prediction2 = "automatic parameter"
    assert not offline_evaluation.exact_match(prediction2, reference)
    assert 0.0 < offline_evaluation.f1_score(prediction2, reference) < 1.0


def test_is_not_found_WHEN_answer_mentions_not_found_THEN_returns_true():
    assert offline_evaluation.is_not_found("Not found")
    assert offline_evaluation.is_not_found("The answer was not found in the document.")
    assert not offline_evaluation.is_not_found("This thesis is about fusion.")


def test_evaluate_sample_WHEN_reference_is_negative_THEN_negative_correct_marks_true():
    results = [
        {
            "text": "No relevant text is available.",
            "doc_id": "doc-1",
            "page": 1,
            "score": 0.0,
        }
    ]

    summary = offline_evaluation.evaluate_sample(
        question="What is the title?",
        reference="Not found",
        results=results,
        prediction="Not found",
    )

    assert summary["is_negative_reference"] is True
    assert summary["is_negative_prediction"] is True
    assert summary["negative_correct"] is True
    assert summary["retrieval_hit"] is None


def test_summarize_metrics_WHEN_examples_are_provided_THEN_aggregates_are_correct():
    examples = [
        {
            "exact_match": True,
            "f1": 1.0,
            "is_negative_reference": False,
            "negative_correct": False,
            "retrieval_hit": True,
            "retrieval_overlap": 1.0,
            "type": "exact",
        },
        {
            "exact_match": False,
            "f1": 0.5,
            "is_negative_reference": True,
            "negative_correct": True,
            "retrieval_hit": None,
            "retrieval_overlap": 0.0,
            "type": "negative",
        },
    ]

    summary = offline_evaluation.summarize_metrics(examples)

    assert summary["count"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["average_f1"] == 0.75
    assert summary["negative_accuracy"] == 1.0
    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["average_retrieval_overlap"] == 0.5


def test_load_qa_dataset_WHEN_file_exists_THEN_parses_examples(tmp_path: Path):
    dataset_path = tmp_path / "qa_dataset.json"
    data = [
        {
            "question": "What is the title?",
            "ground_truth_answer": "Example title",
            "type": "exact",
        }
    ]
    dataset_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = offline_evaluation.load_qa_dataset(dataset_path)
    assert loaded == data
