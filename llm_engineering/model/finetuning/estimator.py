import os
import subprocess

from loguru import logger


class Estimator:
    def __init__(
        self,
        entry_point,
        source_dir,
        hyperparameters,
        requirements_file,
        environment,
    ):
        self.entry_point = entry_point
        self.root_dir = source_dir
        self.requirements_file_path = requirements_file
        self.hyperparameters = hyperparameters
        self.environment = environment

    def setup(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"The directory {self.root_dir} does not exist.")
        if not self.requirements_file_path.exists():
            raise FileNotFoundError(f"The file {self.requirements_file_path} does not exist.")

        command = ["pip", "install", "-r", str(self.requirements_file_path)]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("Failed to install dependencies.")
            logger.error(f"STDERR: {result.stderr}")
        else:
            logger.success("Dependencies installed successfully.")

    def set_environment_variable(self):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = self.environment["HUGGING_FACE_HUB_TOKEN"]
        os.environ["COMET_API_KEY"] = self.environment["COMET_API_KEY"]
        os.environ["COMET_PROJECT_NAME"] = self.environment["COMET_PROJECT_NAME"]

    def fit(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"The directory {self.root_dir} does not exist.")
        entry_point_file_path = self.root_dir / f"llm_engineering/model/finetuning/{self.entry_point}"

        command = [
            "python",
            entry_point_file_path,
            "--num_train_epochs",
            f'{self.hyperparameters["num_train_epochs"]}',
            "--per_device_train_batch_size",
            f'{self.hyperparameters["per_device_train_batch_size"]}',
            "--learning_rate",
            f'{self.hyperparameters["learning_rate"]}',
            "--dataset_huggingface_workspace",
            f'{self.hyperparameters["dataset_huggingface_workspace"]}',
            "--model_output_huggingface_workspace",
            f'{self.hyperparameters["model_output_huggingface_workspace"]}',
            "--is_dummy",
            "True",
            "--finetuning_type",
            "sft",
            # '--output_data_dir', '/path/to/output',
            "--model_dir",
            "kts-temp",
            # '--n_gpus', '1'
        ]

        # subprocess로 실행
        result = subprocess.run(command, capture_output=True, text=True)
        return result
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)
