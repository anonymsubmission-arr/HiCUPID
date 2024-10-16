import os

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]


def should_retry(exception):
    # OSError: [Errno 28] No space left on device
    if isinstance(exception, OSError) and exception.errno == 28:
        return False
    return True


def after_retry(retry_state):
    print(f"({retry_state.attempt_number:02d}/{retry_state.retry_object.stop.max_attempt_number:02d}) Exception: {retry_state.outcome.exception()}")


@retry(stop=stop_after_attempt(20), wait=wait_fixed(10), retry=retry_if_exception(should_retry), after=after_retry)
def snapshot_download_with_retry(*args, **kwargs):
    return snapshot_download(*args, **kwargs)


if __name__ == "__main__":
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    for i, model_id in enumerate(MODELS):
        print(f"({i + 1:02d}/{len(MODELS):02d}) Downloading {model_id}...")
        snapshot_download_with_retry(repo_id=model_id, allow_patterns="model*.safetensors")
