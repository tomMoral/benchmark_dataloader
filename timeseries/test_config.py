import sys
import pytest


def check_test_dataset_get_data(benchmark, dataset_class):
    """Check that the dataset's get_data method returns the expected keys."""
    if dataset_class.name.lower() == "gift-eval":
        pytest.skip("ImageNet dataset requires manual download and setup")


def check_test_solver_run(benchmark, solver_class):
    if (
            solver_class.name.lower() == "pandas-parquet"
            and sys.platform == "darwin"
    ):
        pytest.skip("Pandas parquet dataloader fails to launch on OSX")
