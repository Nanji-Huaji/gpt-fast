import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
import torch
from typing import Dict


class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, torch.Tensor):
            print("Converting tokens from torch.Tensor to list.")
            tokens = tokens.clamp(0, self.processor.vocab_size() - 1)
            tokens = tokens.to(torch.long).flatten().cpu().tolist()
        elif isinstance(tokens, list):
            tokens = list(map(int, filter(lambda x: x >= 0 and x <= self.processor.vocab_size() - 1, tokens)))
        else:
            raise TypeError("Tokens must be a list or torch.Tensor.")
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

    def pad_id(self):
        print(f"Pad ID: {self.processor.pad_id()}")
        pad_id = self.processor.pad_id()
        unk_id = self.processor.unk_id()
        return self.processor.unk_id()

    # @property
    # def special_tokens_map(self):
    #     """
    #     Returns a dictionary mapping special token types to their string representations.
    #     This matches HuggingFace tokenizer's special_tokens_map format.
    #     """
    #     special_tokens = {}

    #     # Add BOS token if it exists
    #     if self.processor.bos_id() != -1:
    #         special_tokens["bos_token"] = "<s>"

    #     # Add EOS token if it exists
    #     if self.processor.eos_id() != -1:
    #         special_tokens["eos_token"] = "</s>"

    #     # Add UNK token if it exists
    #     if self.processor.unk_id() != -1:
    #         special_tokens["unk_token"] = "<unk>"

    #     # Add PAD token if it exists
    #     if self.processor.pad_id() != -1:
    #         special_tokens["pad_token"] = "<pad>"

    #     return special_tokens

    # @property
    # def special_tokens_ids(self):
    #     """
    #     Returns a dictionary mapping special token strings to their IDs.
    #     """
    #     special_tokens = {}

    #     # Add BOS and EOS tokens if they exist
    #     if self.processor.bos_id() != -1:
    #         special_tokens["<s>"] = self.processor.bos_id()
    #     if self.processor.eos_id() != -1:
    #         special_tokens["</s>"] = self.processor.eos_id()

    #     # Add UNK token if it exists
    #     if self.processor.unk_id() != -1:
    #         special_tokens["<unk>"] = self.processor.unk_id()

    #     # Add PAD token if it exists
    #     if self.processor.pad_id() != -1:
    #         special_tokens["<pad>"] = self.processor.pad_id()

    #     return special_tokens

    @property
    def special_tokens_map(self):
        """
        Returns a dictionary mapping special token types to their string representations.
        MODIFIED: This version is tailored for models without a dedicated pad token,
        aliasing the 'unk_token' as the 'pad_token' to meet specific requirements.
        """
        special_tokens = {}
        if self.processor.bos_id() != -1:
            special_tokens["bos_token"] = self.processor.id_to_piece(self.processor.bos_id())
        if self.processor.eos_id() != -1:
            special_tokens["eos_token"] = self.processor.id_to_piece(self.processor.eos_id())
        if self.processor.unk_id() != -1:
            special_tokens["unk_token"] = self.processor.id_to_piece(self.processor.unk_id())
        if "unk_token" in special_tokens:
            special_tokens["pad_token"] = special_tokens["unk_token"]  # This sets 'pad_token': '<unk>'
        print("Special tokens map:", special_tokens)
        return special_tokens

    @property
    def special_tokens_ids(self):
        """
        Returns a dictionary mapping special token strings to their IDs.
        MODIFIED: This version is simplified to reflect that the <unk> token
        is also used for padding.
        """
        special_tokens = {}
        # Add BOS and EOS tokens if they exist
        if self.processor.bos_id() != -1:
            special_tokens[self.processor.id_to_piece(self.processor.bos_id())] = self.processor.bos_id()
        if self.processor.eos_id() != -1:
            special_tokens[self.processor.id_to_piece(self.processor.eos_id())] = self.processor.eos_id()
        # Add UNK token if it exists. This single entry now covers both UNK and PAD roles.
        if self.processor.unk_id() != -1:
            special_tokens[self.processor.id_to_piece(self.processor.unk_id())] = self.processor.unk_id()
        print("Special tokens IDs:", special_tokens)
        return special_tokens


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, self.num_reserved_special_tokens - 5)]
        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    @property
    def special_tokens_map(self):
        """
        Returns a dictionary mapping special token names to their IDs.
        """
        return self.special_tokens

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


def get_tokenizer(tokenizer_model_path, model_name):
    """
    Factory function to get the appropriate tokenizer based on the model name.

    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """

    if "llama-3" in str(model_name).lower():
        return TiktokenWrapper(tokenizer_model_path)
    else:
        return SentencePieceWrapper(tokenizer_model_path)
