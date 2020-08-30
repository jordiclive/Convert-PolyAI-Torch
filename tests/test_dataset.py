import pytest
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader
import os, sys

# dirname, _ = os.path.split(os.path.dirname(__file__))
# path = os.path.join(dirname,'src')
# sys.path.append(dirname)

# to run from terminal , use

# pytest -s tests/test_dataset.py start directory.
from src.config import ConveRTTrainConfig
from src.dataset import load_instances_from_reddit_json

REDDIT_SAMPLE_DATA = {
    "context_author": "Needs_Mega_Magikarp",
    "context/8": "*She giggles at his giggle.* Yay~",
    "context/5": "*He rests his head on yours.*\n\nYou aaaaare. You're the cutest.",
    "context/4": "Pfffft. *She playfully pokes his stomach.* Shuddup.",
    "context/7": "*He hugs you.*\n\nOhmigooods, you're so cute.",
    "context/6": "*She giggles again.* No I'm noooot.",
    "context/1": "*He snorts a laugh*\n\nD'aww. Cute.",
    "context/0": "Meanie.",
    "response_author": "Ironic_Remorse",
    "subreddit": "PercyJacksonRP",
    "thread_id": "2vcitx",
    "context/3": "*He shrugs.*\n\nBut I dun wanna lie!",
    "context": "Cutie.\n\n*He jokes, rubbing your arm again. Vote Craig for best brother 2k15.*",
    "context/2": "*She sticks her tongue out.*",
    "response": "Meanieee. *She pouts.*",
}


@pytest.fixture
def config():
    return ConveRTTrainConfig()


@pytest.fixture
def tokenizer() -> SentencePieceProcessor:
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(config.sp_model_path)
    return tokenizer


def test_load_instances_from_reddit_json(config):
    instances = load_instances_from_reddit_json(config.dataset_path)
    assert len(instances) == 1000


if __name__ == "__main__":
    pytest.main()
