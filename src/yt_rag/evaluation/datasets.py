from __future__ import annotations

from langsmith import Client


def upload_golden_dataset(
    qa_pairs: list[dict],
    dataset_name: str = "yt-rag-golden",
) -> str:
    """Upload a list of Q&A probe dicts to LangSmith as an evaluation dataset.

    Each dict must have at least a "question" key. Optional keys used as
    outputs: "expected_refusal", "answer". All other keys are stored as inputs.

    Returns the dataset ID.
    """
    client = Client()

    existing = list(client.list_datasets(dataset_name=dataset_name))
    for ds in existing:
        client.delete_dataset(dataset_id=ds.id)

    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[
            {
                "question":   q["question"],
                "playlist_id": q.get("playlist_id", ""),
                "user_id":    q.get("user_id", ""),
                "query_type": q.get("query_type", ""),
            }
            for q in qa_pairs
        ],
        outputs=[
            {
                "answer":           q.get("answer", ""),
                "expected_refusal": q.get("expected_refusal", False),
            }
            for q in qa_pairs
        ],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(qa_pairs)} examples to dataset '{dataset_name}' (id={dataset.id})")
    return dataset.id
