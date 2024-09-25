import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/train_config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--model_log_dir",
        type=str,
        default="./model_logs",
        help="Directory where the models are/to be saved",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./data/train",
        help="Directory where the train data is located",
    )
    parser.add_argument(
        "--restore_version",
        type=str,
        default=None,
        help="Version to restore for training continuation",
    )
    return parser.parse_args(args)


class Args:
    def __init__(self):
        self.config_path = "./experiments/config.json"
        self.model_log_dir = "./models/checkpoints"
        self.train_data_dir = "./data/train"
        self.restore_version = None
