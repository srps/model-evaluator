from functools import wraps
import json
import os
from pathlib import Path
import time
from typing import Callable, Tuple, TypeVar

from sentence_transformers import SentenceTransformer

def get_available_st_models() -> list[str]:
    """
    Retrieves a list of available Sentence Transformer models.

    This function scans the 'models' directory in the current working directory
    for subdirectories containing a 'config.json' file. It checks if the config
    file has a 'model_type' key, which should indicate a valid Sentence Transformer model.

    Returns:
        list[str]: A list of relative paths to the available Sentence Transformer models.

    Raises:
        OSError: If there's an error accessing the directory or files.
        json.JSONDecodeError: If there's an error parsing the config.json file.

    Note:
        The function prints error messages to stdout if it encounters issues
        reading or parsing a config file, but continues processing other models.
    """
    models_dir = Path(os.getcwd()) / "models"
    available_models = []
    for root, dirs, files in os.walk(models_dir):
        for dir_name in dirs:
            model_path = Path(root) / dir_name
            config_path = model_path / "config.json"
            if config_path.is_file():
                try:
                    with config_path.open("r") as f:
                        config = json.load(f)
                        if "model_type" in config:
                            available_models.append(model_path.relative_to(models_dir).as_posix())
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Error reading {config_path}: {e}")
    return available_models

def download_model(model_name: str) -> str:
    """
    Downloads and saves a Sentence Transformer model.

    This function downloads a specified Sentence Transformer model,
    saves it to a local directory, and returns the path where the model is saved.

    Args:
        model_name (str): The name of the Sentence Transformer model to download.

    Returns:
        str: The path where the model is saved.

    Raises:
        Any exceptions raised by SentenceTransformer, os.makedirs, or model.save.

    Note:
        This function requires the 'sentence_transformers' library to be installed.
        It also uses Streamlit's 'st.success' function to display a success message,
        so it should be used within a Streamlit application.
    """
    model = SentenceTransformer(model_name)
    model_save_path = os.path.join("models", model_name.replace("/", os.sep))
    os.makedirs(model_save_path, exist_ok=True)
    model.save(model_save_path)
    return model_save_path


T = TypeVar('T')
R = TypeVar('R')

def timing_decorator(func: Callable[..., R]) -> Callable[..., Tuple[R, float]]:
    """
    A decorator that measures the execution time of a function.

    This decorator wraps the given function and returns a new function that,
    when called, will execute the original function and return a tuple containing
    the original function's result and the execution time in seconds.

    Args:
        func (Callable[..., R]): The function to be timed.

    Returns:
        Callable[..., Tuple[R, float]]: A wrapped function that returns a tuple
        containing the original function's result and its execution time.

    Example:
        @timing_decorator
        def example_function(x, y):
            return x + y

        result, execution_time = example_function(3, 4)
        print(f"Result: {result}, Time: {execution_time} seconds")
    """
    @wraps(func)
    def wrapper(*args: T, **kwargs: T) -> Tuple[R, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper
