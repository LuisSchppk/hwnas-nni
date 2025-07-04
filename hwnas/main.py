import json
import os
from search_space.search_space import VGG8ModelSpaceVGG8
from nni.nas.evaluator import FunctionalEvaluator
import nni.nas.strategy as strategy
from nni.nas.experiment import NasExperiment
from nni.nas.experiment.config import NasExperimentConfig
from evaluator.evaluator import hw_evaluation_model
import torch

torch.set_float32_matmul_precision("medium")


def run_evo(id="short_training", output_dir=f"output_{id}", device="cuda"):
    """
    Runs a Neural Architecture Search (NAS) experiment using the Regularized Evolution strategy. This is the experiment detailed in the paper.
    """
    batch_size = 128
    num_workers = 0
    max_epochs = 3
    max_trials = 300
    model_space = VGG8ModelSpaceVGG8()

    search_strategy = strategy.RegularizedEvolution(population_size=100, sample_size=25)
    engine = "sequential"
    evaluator = FunctionalEvaluator(
        hw_evaluation_model,
        **{
            "num_classes": 10,
            "batch_size": batch_size,
            "epochs": max_epochs,
            "num_workers": num_workers,
            "output_csv": os.path.join(output_dir, "results.csv"),
            "csv_suffix": "_evo.",
            "hardware_config": "MNSIM-2.0/SimConfig.ini",
        },
    )

    config = NasExperimentConfig(engine, "simplified", "local", **{"debug": True})
    exp = NasExperiment(model_space, evaluator, search_strategy, config, id=id)

    exp.config.max_trial_number = max_trials
    exp.config.execution_engine.name = engine

    exp.run(debug=True)
    searched_models = exp.export_top_models(formatter="dict", top_k=max_trials)
    with open(os.path.join(output_dir, "top_models.json"), "w") as f:
        json.dump(searched_models, f, indent=4)


def run_tpe(id="short_training", output_dir=f"output_{id}"):
    """
    Runs a Neural Architecture Search (NAS) experiment using the TPE strategy. This is the experiment detailed in the paper.
    """
    batch_size = 128
    num_workers = 0
    max_epochs = 3
    max_trials = 300
    model_space = VGG8ModelSpaceVGG8()

    search_strategy = strategy.TPE()
    engine = "sequential"
    evaluator = FunctionalEvaluator(
        hw_evaluation_model,
        **{
            "num_classes": 10,
            "batch_size": batch_size,
            "epochs": max_epochs,
            "num_workers": num_workers,
            "output_csv": os.path.join(output_dir, "results.csv"),
            "csv_suffix": "_tpe.",
        },
    )

    config = NasExperimentConfig(engine, "simplified", "local", **{"debug": True})

    exp = NasExperiment(model_space, evaluator, search_strategy, config, id=id)

    exp.config.max_trial_number = max_trials
    exp.config.execution_engine.name = engine

    exp.run(debug=True)
    searched_models = exp.export_top_models(formatter="dict", top_k=max_trials)
    with open(os.path.join(output_dir, "top_models.json"), "w") as f:
        json.dump(searched_models, f, indent=4)

    with open(os.path.join(output_dir, "search_stat_dict.json"), "w") as f:
        json.dump(search_strategy.state_dict(), f, indent=4)


def main():
    torch.set_float32_matmul_precision("medium")
    id = "short_training"
    output_dir = f"output_{id}"
    os.makedirs(output_dir, exist_ok=True)
    run_evo(id, output_dir)

    # Optional: Run TPE Search. Results were worse than with evo.
    # source = Path(output_dir)
    # destination = Path(output_dir + "_evo")
    # shutil.copytree(source, destination)
    # run_tpe(id, output_dir)


if __name__ == "__main__":
    main()
