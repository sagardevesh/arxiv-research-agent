import pytest
from unittest.mock import MagicMock, patch

from eval.harness import THRESHOLDS, check_thresholds, load_questions, run_eval


def test_load_questions_returns_list_of_dicts():
    questions = load_questions()
    assert isinstance(questions, list)
    assert len(questions) > 0
    for q in questions:
        assert "question" in q
        assert "ground_truth" in q


def test_check_thresholds_all_pass(capsys):
    scores = {k: v + 0.1 for k, v in THRESHOLDS.items()}
    result = check_thresholds(scores)
    assert result is True
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert "FAIL" not in captured.out


def test_check_thresholds_one_fail(capsys):
    scores = {k: v + 0.1 for k, v in THRESHOLDS.items()}
    first_metric = next(iter(THRESHOLDS))
    scores[first_metric] = THRESHOLDS[first_metric] - 0.1
    result = check_thresholds(scores)
    assert result is False
    assert "FAIL" in capsys.readouterr().out


def test_run_eval_uses_injected_fns():
    """run_eval should call the injected retriever and answer functions."""
    retriever_calls = []
    answer_calls = []

    def mock_retriever(question):
        retriever_calls.append(question)
        return ["Transformers use self-attention to model sequence dependencies."]

    def mock_answer(question, contexts):
        answer_calls.append((question, contexts))
        return "Self-attention allows transformers to relate positions in the sequence."

    mock_ragas_llm = MagicMock()
    mock_metric = MagicMock()
    mock_metric.name = "faithfulness"
    mock_result = {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_precision": 0.75}

    with patch("eval.harness.evaluate", return_value=mock_result) as mock_eval, \
         patch("eval.harness.Faithfulness", return_value=mock_metric), \
         patch("eval.harness.AnswerRelevancy", return_value=mock_metric), \
         patch("eval.harness.ContextPrecision", return_value=mock_metric), \
         patch("eval.harness.EvaluationDataset") as mock_dataset, \
         patch("eval.harness.SingleTurnSample"):
        mock_dataset.return_value = MagicMock()
        scores = run_eval(
            retriever_fn=mock_retriever,
            answer_fn=mock_answer,
            ragas_llm=mock_ragas_llm,
        )

    questions = load_questions()
    assert len(retriever_calls) == len(questions)
    assert len(answer_calls) == len(questions)
    assert scores["faithfulness"] == 0.85
    assert scores["answer_relevancy"] == 0.80
    assert scores["context_precision"] == 0.75
