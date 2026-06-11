from __future__ import annotations

import json
import pathlib

from langsmith import Client


def upload_golden_dataset(
    qa_path: str | pathlib.Path,
    dataset_name: str = "yt-rag-golden",
) -> str:
    """Upload a golden Q&A JSON file to LangSmith as an evaluation dataset.

    JSON format: list of {"question": str, "answer": str, "playlist_id"?: str}

    Returns the dataset ID.
    """
    client = Client()
    path = pathlib.Path(qa_path)
    qa_pairs: list[dict] = json.loads(path.read_text(encoding="utf-8"))

    # Delete existing dataset with same name to allow clean re-uploads
    existing = list(client.list_datasets(dataset_name=dataset_name))
    for ds in existing:
        client.delete_dataset(dataset_id=ds.id)

    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": q["question"], "playlist_id": q.get("playlist_id", "")} for q in qa_pairs],
        outputs=[{"answer": q["answer"]} for q in qa_pairs],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(qa_pairs)} examples to dataset '{dataset_name}' (id={dataset.id})")
    return dataset.id
