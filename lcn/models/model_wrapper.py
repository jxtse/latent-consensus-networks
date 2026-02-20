# lcn/models/model_wrapper.py
"""Model wrapper for LCN."""

from typing import Dict, List, Optional, Tuple

import torch

from lcn.core.kv_cache import KVCache


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
        # Cache for latent realignment matrices: model_id -> (matrix, target_norm)
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

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

    @staticmethod
    def _past_length(past_key_values: Optional[KVCache]) -> int:
        """
        Get the sequence length from a KV-Cache.

        Args:
            past_key_values: KV-Cache tuple of (key, value) per layer,
                where key/value have shape [batch, num_heads, seq_len, head_dim]

        Returns:
            Sequence length, or 0 if cache is None/empty
        """
        if not past_key_values:
            return 0
        # First layer, first tensor (key), dimension -2 is seq_len
        k = past_key_values[0][0]
        return k.shape[-2]

    def _build_latent_realign_matrix(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the latent space realignment matrix from model embeddings.

        The realignment matrix transforms hidden states back to the input embedding
        space, enabling effective latent reasoning steps.

        Args:
            model: The language model
            device: Device to place tensors on

        Returns:
            Tuple of (realign_matrix, target_norm)
            - realign_matrix: [hidden_dim, hidden_dim] transformation matrix
            - target_norm: Target norm for normalized output

        Raises:
            RuntimeError: If model embeddings are not accessible
        """
        input_embeds = (
            model.get_input_embeddings()
            if hasattr(model, "get_input_embeddings")
            else None
        )
        output_embeds = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else None
        )

        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)

        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )

        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

        # Solve least squares: find M such that output_weight @ M ≈ input_weight
        # This is (O^T O) M = O^T I, solved via torch.linalg.solve
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)

        target_norm = input_weight.norm(dim=1).mean().detach()

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get or create the latent realignment matrix for a model.

        Caches the matrix per model instance to avoid recomputation.

        Args:
            model: The language model
            device: Device to place tensors on

        Returns:
            Tuple of (realign_matrix, target_norm)
        """
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(
        self,
        hidden: torch.Tensor,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Apply latent space realignment to a hidden state.

        Transforms the hidden state and normalizes to target embedding norm.

        Args:
            hidden: Hidden state tensor [batch, hidden_dim]
            model: The language model (used to get/cache realignment matrix)

        Returns:
            Aligned hidden state [batch, hidden_dim]
        """
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)

        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        latent_steps: int,
        past_key_values: Optional[KVCache] = None,
    ) -> Tuple[KVCache, torch.Tensor]:
        """
        Generate KV-Cache and hidden state through latent steps.

        Performs an initial forward pass to get the KV-Cache, then optionally
        performs additional latent steps where the hidden state is fed back
        as input embeddings.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len], or None to use all ones
            latent_steps: Number of additional latent reasoning steps
            past_key_values: Optional existing KV-Cache to continue from

        Returns:
            Tuple of (kv_cache, hidden_state)
            - kv_cache: Updated KV-Cache after all steps
            - hidden_state: Final hidden state [batch, hidden_dim]

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input_ids is not 2D
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before generate_latent().")

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        # Extend attention mask for past key values
        if past_key_values is not None:
            past_len = self._past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # Initial forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

        # Latent steps
        for _ in range(latent_steps):
            # Get embedding for next latent step
            if self.latent_space_realign:
                latent_vec = self._apply_latent_realignment(last_hidden, self.model)
            else:
                # Use hidden state directly as input embedding
                latent_vec = last_hidden

            latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]

            # Update attention mask for new position
            past_len = self._past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )

            # Forward pass with embedding input
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

        return past, last_hidden

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[KVCache] = None,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Generate text from current state.

        Uses the model's generate() method to produce text continuations.
        When past_key_values is provided, extends the attention mask and
        sets cache_position for proper continuation.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            past_key_values: Optional KV-Cache from previous latent steps
            max_new_tokens: Maximum number of tokens to generate (default: 256)

        Returns:
            List of generated text strings, one per batch item

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input_ids is not 2D
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before generate_text().")

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        # Track prompt lengths for stripping from output
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        # Handle past_key_values: extend attention mask and set cache_position
        cache_position = None
        if past_key_values is not None:
            past_len = self._past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # Generate text
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        # Decode generated sequences, stripping prompt tokens
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            generations.append(text)

        return generations
