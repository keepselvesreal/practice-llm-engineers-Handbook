from pathlib import Path

import pytest
from huggingface_hub import HfApi

from llm_engineering.model.finetuning.estimator import Estimator
from llm_engineering.settings import settings


@pytest.fixture(scope="class")
def estimator():
    root_dir = Path(__file__).resolve().parent.parent.parent
    finetuning_requirements_path = root_dir / "llm_engineering/model/finetuning"
    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    hyperparameters = {
        "finetuning_type": "sft",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "learning_rate": 3e-4,
        "dataset_huggingface_workspace": "mlabonne",
        "model_output_huggingface_workspace": huggingface_user,
    }
    estimator = Estimator(
        entry_point="finetune.py",
        source_dir=root_dir,
        hyperparameters=hyperparameters,
        requirements_file=finetuning_requirements_path,
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": "dummy",
            "COMET_PROJECT_NAME": "dummy",
            # "COMET_API_KEY": settings.COMET_API_KEY,
            # "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
    )
    yield estimator


# @pytest.fixture(scope="session", autouse=True) # todo 이것 없으면?
# def setup_environment():

#     print(f"Python Path: {sys.executable}")
#     print(f"Torch Version: {torch.__version__}")

#     subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
#     subprocess.check_call([
#         sys.executable,
#         "-m",
#         "pip",
#         "install",
#         "torch==2.4.0"
#     ])

#     finetuning_requirements_path ="/home/ubuntu/LLM-Engineers-Handbook/llm_engineering/model/finetuning/requirements.txt"
#     if not os.path.exists(finetuning_requirements_path):
#         raise FileNotFoundError(f"Requirements file not found: {finetuning_requirements_path}")

#     try:
#         result = subprocess.run(
#             [sys.executable, "-m", "pip", "install", "-r", finetuning_requirements_path,  "--no-cache-dir"],
#             capture_output=True,
#             text=True
#         )
#         print(f"STDOUT: {result.stdout}")
#         print(f"STDERR: {result.stderr}")
#         result.check_returncode() # todo ??
#     except subprocess.CalledProcessError as e:
#         print(f"Exit code: {e.returncode}")
#         print(f"STDOUT: {e.stdout if hasattr(e, 'stdout') else 'N/A'}")
#         print(f"STDERR: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
#         raise
