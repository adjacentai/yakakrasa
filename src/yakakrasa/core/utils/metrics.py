from typing import List, Dict, Tuple


def compute_intent_accuracy(true_labels: List[str], pred_labels: List[str]) -> float:
    assert len(true_labels) == len(pred_labels)
    if not true_labels:
        return 0.0
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    return correct / len(true_labels)


def compute_entity_pr(
    true_entities: List[List[Dict]],
    pred_entities: List[List[Dict]],
) -> Tuple[float, float, float]:
    """
    Very simple entity metrics based on exact match of (text, type) pairs.
    true_entities/pred_entities is a list (per sample) of dicts with keys: text, entity (type).
    Returns precision, recall, f1.
    """
    tp = 0
    fp = 0
    fn = 0

    for gold, pred in zip(true_entities, pred_entities):
        gold_set = {(e.get("text", ""), e.get("entity") or e.get("type")) for e in gold}
        pred_set = {(e.get("text", ""), e.get("entity") or e.get("type")) for e in pred}

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
