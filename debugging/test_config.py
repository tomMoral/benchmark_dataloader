import pytest


def check_test_dataset_get_data(benchmark, dataset_class):
    """Check that the dataset's get_data method returns the expected keys."""
    if dataset_class.name.lower() == "openbhb":
        pytest.skip("OpenBHB dataset requires nidl and downloaded data")
