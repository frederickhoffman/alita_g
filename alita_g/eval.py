import argparse

from datasets import load_dataset
from dotenv import load_dotenv

from alita_g.graph import AlitaGAgent
from alita_g.monitoring import WandBMonitor

load_dotenv()

def run_eval(dataset_name: str, split: str, num_samples: int = None, pass_n: int = 1):
    print(f"Running {dataset_name} evaluation on {split} set (pass@{pass_n})...")

    # Load dataset
    try:
        if dataset_name.lower() == "gaia":
            dataset = load_dataset("gaia-benchmark/GAIA", "2023", split=split)
        elif dataset_name.lower() == "pathvqa":
            # Assuming standard pathvqa or medical-vqa/pathvqa
            dataset = load_dataset("flaviagiammarino/path-vqa", split=split)
        elif dataset_name.lower() == "hle":
            # Humanity's Last Exam
            dataset = load_dataset("cais/hle", split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}. Falling back to a dummy eval.")
        dataset = [{"question": "What is 2+2?", "Answer": "4"}]

    monitor = WandBMonitor()
    monitor.start_run(
        name=f"gaia-{split}-pass{pass_n}",
        config={
            "dataset": dataset_name,
            "split": split,
            "pass_n": pass_n,
            "model": "gpt-4o",
            "threshold": 0.7
        }
    )

    agent = AlitaGAgent()
    compiled_graph = agent.build_graph()

    correct = 0
    total = 0

    samples = dataset
    if num_samples:
        samples = dataset.select(range(min(num_samples, len(dataset))))

    for i, row in enumerate(samples):
        question = row.get("question", row.get("Task", ""))
        ground_truth = row.get("Answer", row.get("Final answer", ""))

        print(f"Task {i+1}/{len(samples)}: {question[:100]}...")

        # Simple implementation of pass@n
        success = False
        for _ in range(pass_n):
            result = compiled_graph.invoke({"messages": [("user", question)]})
            prediction = result["messages"][-1].content

            # Basic matching (GAIA evaluation is usually more complex, e.g., using an LLM judge)
            if ground_truth.strip().lower() in prediction.strip().lower():
                success = True
                break

        if success:
            correct += 1
        total += 1

        # Log individual task performance
        monitor.log_metrics({
            "task_id": i,
            "correct": success,
            "cumulative_accuracy": correct / total
        })

    final_accuracy = correct / total
    print(f"Final Accuracy: {final_accuracy:.2%}")

    monitor.log_metrics({"final_accuracy": final_accuracy})
    monitor.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="GAIA", help="Dataset to evaluate (GAIA, PathVQA, HLE)"
    )
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument(
        "--samples", type=int, default=None, help="Number of samples to evaluate"
    )
    parser.add_argument("--pass_n", type=int, default=1, help="N for pass@N")

    args = parser.parse_args()
    run_eval(args.dataset, args.split, args.samples, args.pass_n)
