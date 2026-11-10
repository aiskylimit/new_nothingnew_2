import json

from eval.summarize_results import result_row


def test_result_row_joins_evaluation_with_checkpoint_token_count(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoint-25"
    checkpoint.mkdir(parents=True)
    (checkpoint / "follow_the_leader_state.json").write_text(
        json.dumps({"cumulative_generated_tokens": 12345}),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "base_model": "Qwen/Qwen3-4B",
                "checkpoint_dir": str(checkpoint),
                "dataset": "aime24",
                "seed": 42,
                "average_at_n_pct": 55.0,
            }
        ),
        encoding="utf-8",
    )

    row = result_row(result_path)

    assert row["checkpoint_step"] == 25
    assert row["cumulative_training_tokens"] == 12345
    assert row["average_at_n_pct"] == 55.0
