import os
from pathlib import Path

import pkg_resources
from loguru import logger

from llm_engineering.settings import settings


class TestEstimator:
    def test_setup(self, estimator):
        estimator.setup()

        file_path = Path("/home/ubuntu/LLM-Engineers-Handbook/llm_engineering/model/finetuning/requirements.txt")
        with file_path as f:
            required = f.read().splitlines()

        installed = [pkg.key for pkg in pkg_resources.working_set]

        for package in required:
            package_name = package.split("==")[0]
            assert package_name.lower() in installed

    def test_set_environment_variable(self, estimator):
        initial_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        estimator.set_environment_variable()

        current_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        assert initial_token != current_token
        assert current_token == settings.HUGGINGFACE_ACCESS_TOKEN

    def test_fit(self, estimator):
        result = estimator.fit()
        logger.info("=======Test method에서의 출력=======")
        logger.info("STDOUT:", result.stdout)
        logger.info("STDERR:", result.stderr)
