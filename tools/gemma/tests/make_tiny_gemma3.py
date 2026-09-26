"""Builds a tiny random Gemma 3 checkpoint in Hugging Face format for the offline G0 smoke test.

Same layout as google/gemma-3-4b-it-qat-*: Gemma3ForConditionalGeneration (text + a tiny vision tower),
BF16 safetensors, SentencePiece tokenizer.model with <start_of_turn>/<end_of_turn>, and a Gemma-3-style
chat template in tokenizer_config.json. The weights are random: it only tests the pipeline's plumbing.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# Stand-in for the Gemma 3 chat template (same structure: bos, system text folded into the first user turn,
# trimmed contents, 'model' role, generation prompt). Only used by the smoke test.
CHAT_TEMPLATE = r"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""


def build(out_dir: Path, text_file: Path, vocab_size: int = 8000, seed: int = 0) -> Path:
    import sentencepiece as spm
    import torch
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    prefix = out_dir / "spm"
    spm.SentencePieceTrainer.train(
        input=str(text_file), model_prefix=str(prefix), vocab_size=vocab_size, model_type="bpe",
        character_coverage=1.0, byte_fallback=True, add_dummy_prefix=False, split_digits=True,
        remove_extra_whitespaces=False, normalization_rule_name="identity", allow_whitespace_only_pieces=True,
        pad_id=0, eos_id=1, bos_id=2, unk_id=3, pad_piece="<pad>", eos_piece="<eos>", bos_piece="<bos>",
        unk_piece="<unk>", user_defined_symbols=["<start_of_turn>", "<end_of_turn>"], hard_vocab_limit=False,
        minloglevel=2)
    (out_dir / "spm.model").rename(out_dir / "tokenizer.model")
    (out_dir / "spm.vocab").unlink()
    sp = spm.SentencePieceProcessor(model_file=str(out_dir / "tokenizer.model"))
    n_vocab = sp.get_piece_size()

    torch.manual_seed(seed)
    text_cfg = Gemma3TextConfig(
        vocab_size=n_vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, max_position_embeddings=8192, sliding_window=16,
        rope_theta=1_000_000.0, rope_local_base_freq=10_000.0, query_pre_attn_scalar=32, rms_norm_eps=1e-6,
        pad_token_id=0, eos_token_id=1, bos_token_id=2)
    vision_cfg = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                    image_size=32, patch_size=8)
    cfg = Gemma3Config(text_config=text_cfg, vision_config=vision_cfg, mm_tokens_per_image=4,
                       boi_token_index=n_vocab - 3, eoi_token_index=n_vocab - 2, image_token_index=n_vocab - 1)
    model = Gemma3ForConditionalGeneration(cfg).to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)

    specials = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>"}
    (out_dir / "special_tokens_map.json").write_text(json.dumps(specials, indent=2), encoding="utf-8")
    tok_cfg = dict(specials, add_bos_token=True, add_eos_token=False, tokenizer_class="GemmaTokenizer",
                   model_max_length=8192, chat_template=CHAT_TEMPLATE)
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")
    return out_dir
