import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BASE_CONFIG = {
    "hidden_dim": 128,
    "num_layers": 3,
    "mlp_layers": 2,
    "learning_rate": 0.001,
    "decay_rate": 1.2,
    "max_epochs": 30,
}

SWEEPS = {
    "decay_rate": [1.0, 1.2, 1.5],
    "max_epochs": [20, 30, 40],
}


def build_one_factor_configs():
    configs = []
    seen = set()
    for factor, values in SWEEPS.items():
        for value in values:
            config = dict(BASE_CONFIG)
            config[factor] = value
            key = tuple(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append((factor, config))
    return configs


def command_for_config(args, factor, config):
    suffix = (
        f"{args.results_suffix_prefix}_sens_"
        f"{factor}_"
        f"hd{config['hidden_dim']}_"
        f"nl{config['num_layers']}_"
        f"mlp{config['mlp_layers']}_"
        f"lr{config['learning_rate']}_"
        f"dr{config['decay_rate']}_"
        f"ep{config['max_epochs']}"
    )
    cmd = [
        sys.executable,
        str(Path(args.training_script)),
        "--time_limit",
        str(args.time_limit),
        "--node_limit",
        str(args.node_limit),
        "--hidden_dim",
        str(config["hidden_dim"]),
        "--num_layers",
        str(config["num_layers"]),
        "--mlp_layers",
        str(config["mlp_layers"]),
        "--learning_rate",
        str(config["learning_rate"]),
        "--decay_rate",
        str(config["decay_rate"]),
        "--max_epochs",
        str(config["max_epochs"]),
        "--decay_every",
        str(args.decay_every),
        "--rank_method",
        args.rank_method,
        "--cost_loss",
        args.cost_loss,
        "--rank_loss",
        args.rank_loss,
        "--loss_weight",
        str(args.loss_weight),
        "--train_ids_json",
        args.train_ids_json,
        "--test_ids_json",
        args.test_ids_json,
        "--data_root",
        args.data_root,
        "--results_root",
        args.results_root,
        "--device",
        args.device,
        "--results_suffix",
        suffix,
    ]
    if args.instance_training_dir is not None:
        cmd.extend(["--instance_training_dir", args.instance_training_dir])
    if args.instance_testing_dir is not None:
        cmd.extend(["--instance_testing_dir", args.instance_testing_dir])
    if args.execution_dir is not None:
        cmd.extend(["--execution_dir", args.execution_dir])
    if args.valid_ratio is not None:
        cmd.extend(["--valid_ratio", str(args.valid_ratio)])
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Launch a small one-factor-at-a-time hyperparameter sensitivity check. "
            "Defaults print commands only; pass --execute to run them."
        )
    )
    parser.add_argument("--training_script", default="main.py")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--time_limit", type=float, default=10)
    parser.add_argument("--node_limit", type=int, default=1000)
    parser.add_argument("--rank_method", default="dense")
    parser.add_argument("--cost_loss", default="MAE")
    parser.add_argument("--rank_loss", default="LambdaRank")
    parser.add_argument("--loss_weight", type=float, default=0.5)
    parser.add_argument("--decay_every", type=int, default=5)
    parser.add_argument("--valid_ratio", type=float, default=None)
    parser.add_argument("--train_ids_json", default="data/filtered_ids_10s_node_le_1000_stratified_allscarce_700_300_training.json")
    parser.add_argument("--test_ids_json", default="data/filtered_ids_10s_node_le_1000_stratified_allscarce_700_300_testing.json")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--instance_training_dir", default=None)
    parser.add_argument("--instance_testing_dir", default=None)
    parser.add_argument("--execution_dir", default=None)
    parser.add_argument("--results_root", default="outputs/hparam_sensitivity")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--results_suffix_prefix", default="_10s_MAE_LambdaRank")
    args = parser.parse_args()

    configs = build_one_factor_configs()
    print(f"Prepared {len(configs)} one-factor sensitivity runs.")
    for index, (factor, config) in enumerate(configs, start=1):
        cmd = command_for_config(args, factor, config)
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"\n[{index}/{len(configs)}] {factor}: {config}")
        print(printable)
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
