import json
import torch

from modules.translator_model import TranslatorModel
from utils.vocab import Vocabulary


class GlossTranslator:
    def __init__(self, model_path: str, vocab_path: str, device: str = "cpu"):
        # Loading checkpoint
        self.checkpoint = torch.load(
            model_path, weights_only=False, map_location=torch.device(device)
        )

        self.config = self.checkpoint["config"]

        # Loads vocabulary for glosses and sentences in Phoenix-T dataset
        vocab = json.load(open(vocab_path))
        glosses, texts = vocab["glosses"], vocab["words"]
        self.gloss_vocab = Vocabulary(glosses[2:])
        self.text_vocab = Vocabulary(texts[3:])

        # Creates translator model based on configuration dictionary stored in checkpoint
        self.model = TranslatorModel(
            src_vocab=self.gloss_vocab,
            trg_vocab=self.text_vocab,
            d_model=self.config.get("d_model", 512),
            hidden_size=self.config.get("hidden_size", 2048),
            heads=self.config.get("heads", 8),
            num_encoders=self.config.get("num_encoders", 2),
            num_decoders=self.config.get("num_decoders", 2),
            dropout=self.config.get("dropout", 0.1),
            norm_first=self.config.get("norm_first", False),
        )

        # Load states in dictionary
        self.model.load_state_dict(self.checkpoint["model_state_dict"])

    def translate(self, sentence: str):
        # Tokenizes initial sentence
        tokens = self.gloss_vocab.tokenize(sentence).unsqueeze(0)

        # Use Translator's greedy decoder to translate gloss into German sentence
        translated_tokens = self.model.greedy_decode(tokens)

        # Convert tensor of tokens back into a string
        return self.text_vocab.decode_batch(translated_tokens.tolist())[0]
    