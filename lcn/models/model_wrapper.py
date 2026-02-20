# lcn/models/model_wrapper.py
"""Model wrapper for LCN."""

from typing import Dict, List, Optional, Tuple

import torch


class LCNModelWrapper:
    """
    Wrapper for language models in Latent Consensus Networks.

    Handles model/tokenizer management and input preparation.
    Model and tokenizer are not loaded on init - call load() to load them.

    Args:
        model_name: HuggingFace model name or path
        device: Device to use for computation
        latent_space_realign: Whether to apply latent space realignment
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        latent_space_realign: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.latent_space_realign = latent_space_realign
        self.tokenizer = None
        self.model = None

    def render_chat(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> str:
        """
        Render chat messages to a prompt string.

        Uses the tokenizer's chat_template if available, otherwise falls back
        to a simple format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add assistant prompt at end

        Returns:
            Formatted prompt string
        """
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

        # Fallback format
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize chat messages for model input.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add assistant prompt at end

        Returns:
            Tuple of (input_ids, attention_mask) tensors on the wrapper's device
        """
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return input_ids, attention_mask
