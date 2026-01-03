import json
import os
import torch
import torch.nn as nn
import gc
import random
import shutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    EncoderDecoderModel,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderConfig
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

# ------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------
SEED = 42
NUM_STAGES_PER_CYCLE = 5
NUM_CYCLES = 3
EPOCHS_PER_STAGE = 3

DATA_FILE = "train.jsonl"
OUTPUT_ROOT = "./final_stable_pipeline"

# Inference Controls
RUN_INFERENCE_EVERY_STAGE = False
RUN_INFERENCE_END_OF_CYCLE = True

# --- MODEL OPTIONS WITH EXPLICIT USE_FAST ---
ENCODER_OPTIONS = [
    {"model": "jhu-clsp/mmBERT-small",            "type": "enc-only", "use_fast": True},
    #{"model": "tencent/HY-MT1.5-1.8B",            "type": "dec-only", "use_fast": True},
    {"model": "Qwen/Qwen2.5-0.5B",                "type": "dec-only", "use_fast": True},
    {"model": "facebook/nllb-200-distilled-600M", "type": "seq2seq",  "use_fast": False},
    {"model": "facebook/m2m100_418M",             "type": "seq2seq",  "use_fast": False}
]

DECODER_OPTIONS = [
    {"model": "jhu-clsp/mmBERT-small",             "type": "enc-only", "use_fast": True},
    {"model": "bert-base-multilingual-cased",      "type": "enc-only", "use_fast": True},
    #{"model": "tencent/HY-MT1.5-1.8B",             "type": "dec-only", "use_fast": True},
    {"model": "Qwen/Qwen2.5-0.5B",                 "type": "dec-only", "use_fast": True},
    {"model": "facebook/nllb-200-distilled-600M",  "type": "seq2seq",  "use_fast": False},
    {"model": "facebook/m2m100_418M",              "type": "seq2seq",  "use_fast": False}
]

# Create Dummy Data
if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
print(f"📝 Initializing {DATA_FILE}...")
with open(DATA_FILE, "w") as f:
    f.write(json.dumps({"source": "Hello world", "target": "Hallo Welt", "lang_pair": "en_US-de_DE"}) + "\n")
    f.write(json.dumps({"source": "AI is evolving", "target": "KI entwickelt sich", "lang_pair": "en_US-de_DE"}) + "\n")
    f.write(json.dumps({"source": "Guten Tag", "target": "Good day", "lang_pair": "de_DE-en_US"}) + "\n")

TEST_SENTENCES = ["Hello world", "Artificial Intelligence is the future"]

# ------------------------------------------------------------------------
# 2. HELPERS: LANG MAPPING & PATCHES
# ------------------------------------------------------------------------
def map_language_code(model_id, raw_code):
    base_lang = raw_code.split("_")[0]
    if "nllb" in model_id:
        mapping = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "zh": "zho_Hans"}
        return mapping.get(base_lang, "eng_Latn")
    elif "m2m" in model_id:
        return base_lang
    return None

def patch_maskedlm_forward(decoder_model):
    original_forward = decoder_model.forward
    def forward(*args, **kwargs):
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        kwargs.pop("num_items_in_batch", None)
        outputs = original_forward(*args, **kwargs)
        return CausalLMOutputWithCrossAttentions(
            loss=outputs.loss, logits=outputs.logits, past_key_values=None,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            cross_attentions=getattr(outputs, "cross_attentions", None)
        )
    decoder_model.forward = forward
    return decoder_model

def patch_causallm_forward(decoder_model):
    original_forward = decoder_model.forward
    def forward(*args, **kwargs):
        kwargs.pop("encoder_hidden_states", None)
        kwargs.pop("encoder_attention_mask", None)
        kwargs.pop("num_items_in_batch", None)
        return original_forward(*args, **kwargs)
    decoder_model.forward = forward
    return decoder_model

def patch_seq2seq_decoder_forward(full_decoder_model):
    decoder = full_decoder_model.get_decoder()
    lm_head = full_decoder_model.lm_head
    original_forward = decoder.forward

    def forward(*args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        decoder_outputs = original_forward(*args, **kwargs)
        
        # Manually apply the language modeling head to get logits
        logits = lm_head(decoder_outputs.last_hidden_state)

        return CausalLMOutputWithCrossAttentions(
            loss=None, # Loss is calculated by the wrapping EncoderDecoderModel
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )
    
    decoder.forward = forward
    return decoder

# ------------------------------------------------------------------------
# 3. ROBUST MODEL CLASS
# ------------------------------------------------------------------------
class RobustEncoderDecoderModel(EncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        super(EncoderDecoderModel, self).__init__(config)
        self.encoder, self.decoder = encoder, decoder
        self.decoder.config.is_decoder = True
        self.decoder.config.add_cross_attention = True
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
            init_std = getattr(self.decoder.config, "initializer_range", 0.02)
            self.enc_to_dec_proj.weight.data.normal_(mean=0.0, std=init_std)
            if self.enc_to_dec_proj.bias is not None: self.enc_to_dec_proj.bias.data.zero_()
        else:
            self.enc_to_dec_proj = None

# ------------------------------------------------------------------------
# 4. CURRICULUM GENERATION
# ------------------------------------------------------------------------
def generate_stages(seed, count):
    random.seed(seed)
    stages = []
    print(f"\n🎲 Generating randomized curriculum (Seed {seed}):")
    for i in range(1, count + 1):
        enc = random.choice(ENCODER_OPTIONS)
        dec = random.choice(DECODER_OPTIONS)
        stage = {
            "name": f"stage{i}", "enc": enc["model"], "enc_type": enc["type"],
            "enc_use_fast": enc.get("use_fast", True),
            "dec": dec["model"], "dec_type": dec["type"]
        }
        stages.append(stage)
        print(f"   [{stage['name']}] Enc: {stage['enc']} (Fast:{stage['enc_use_fast']}) -> Dec: {stage['dec']}")
    return stages

GENERATED_STAGES = generate_stages(SEED, NUM_STAGES_PER_CYCLE)

# ------------------------------------------------------------------------
# 5. SMART WEIGHT LOADER
# ------------------------------------------------------------------------
def smart_load_weights(model, adapter_path, load_enc, load_dec, load_embed):
    print(f"♻️  Loading weights from {adapter_path}...")
    sf_file = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_file = os.path.join(adapter_path, "adapter_model.bin")
    state_dict = load_file(sf_file) if os.path.exists(sf_file) else (torch.load(bin_file, map_location="cpu") if os.path.exists(bin_file) else None)
    
    if not state_dict: return
    filtered = {k: v for k, v in state_dict.items() if (("embed_tokens" in k or "lm_head" in k) and load_embed) or ("encoder" in k and load_enc) or ("decoder" in k and load_dec)}
    if filtered: 
        model.load_state_dict(filtered, strict=False)
        print(f"   ✅ Transferred {len(filtered)} tensors.")

# ------------------------------------------------------------------------
# 6. TRAINING ENGINE
# ------------------------------------------------------------------------
def run_stage(cycle_num, stage_info, output_dir, prev_info, do_inference=True):
    enc_id, enc_type, use_fast = stage_info['enc'], stage_info['enc_type'], stage_info['enc_use_fast']
    dec_id, dec_type = stage_info['dec'], stage_info['dec_type']
    
    print(f"\n🚀 CYCLE {cycle_num} | {stage_info['name'].upper()}")
    print(f"   Tokenizer: {enc_id} (use_fast={use_fast})")

    # --- TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(enc_id, trust_remote_code=True, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # --- MODELS ---
    if enc_type == "seq2seq":
        full_enc = AutoModel.from_pretrained(enc_id, trust_remote_code=True)
        encoder = full_enc.get_encoder() if hasattr(full_enc, "get_encoder") else full_enc.model.encoder
    else:
        encoder = AutoModel.from_pretrained(enc_id, trust_remote_code=True)

    if dec_type == "seq2seq":
        full_dec = AutoModelForSeq2SeqLM.from_pretrained(dec_id, trust_remote_code=True)
        decoder = patch_seq2seq_decoder_forward(full_dec)
    elif dec_type == "enc-only":
        decoder = patch_maskedlm_forward(AutoModelForMaskedLM.from_pretrained(dec_id, trust_remote_code=True))
    else:
        decoder = patch_causallm_forward(AutoModelForCausalLM.from_pretrained(dec_id, trust_remote_code=True))

    decoder.resize_token_embeddings(len(tokenizer))
    model = RobustEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Config
    start_id = tokenizer.cls_token_id if tokenizer.cls_token_id else (tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id)
    model.config.decoder_start_token_id, model.config.pad_token_id = start_id, tokenizer.pad_token_id
    model.config.vocab_size, model.config.eos_token_id = len(tokenizer), tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id
    model.config.max_length, model.config.num_beams, model.config.use_cache = 128, 4, False
    model.config.gradient_checkpointing = True

    # --- PEFT ---
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64, lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense", "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"] 
    )
    model = get_peft_model(model, peft_config)

    if prev_info:
        smart_load_weights(model, prev_info['path'], (prev_info['enc']==enc_id), (prev_info['dec']==dec_id), (prev_info['enc']==enc_id))

    # --- DATA ---
    dataset = load_dataset("json", data_files=DATA_FILE, split="train", download_mode="force_redownload")
    def preprocess(batch):
        ids, masks, labels = [], [], []
        for i in range(len(batch["source"])):
            src_raw, tgt_raw = batch["lang_pair"][i].split("-")
            sc, tc = map_language_code(enc_id, src_raw), map_language_code(enc_id, tgt_raw)
            if sc and hasattr(tokenizer, "src_lang"): tokenizer.src_lang = sc
            if tc and hasattr(tokenizer, "tgt_lang"): tokenizer.tgt_lang = tc
            
            s = tokenizer(batch["source"][i], max_length=128, truncation=True)
            t = tokenizer(text_target=batch["target"][i], max_length=128, truncation=True)
            ids.append(s["input_ids"]); masks.append(s["attention_mask"]); labels.append(t["input_ids"])
        return {"input_ids": ids, "attention_mask": masks, "labels": labels}

    tokenized = dataset.map(preprocess, batched=True, batch_size=8, remove_columns=dataset.column_names)

    # --- TRAIN ---
    trainer = Seq2SeqTrainer(
        model=model, train_dataset=tokenized, tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        args=Seq2SeqTrainingArguments(output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=2e-4, num_train_epochs=EPOCHS_PER_STAGE, fp16=True if torch.cuda.is_available() else False, save_strategy="no", report_to="none")
    )
    trainer.train()
    model.save_pretrained(output_dir)
    
    if do_inference:
        print(f"\n🧪 INFERENCE TEST:"); model.eval(); model.config.use_cache = True
        for text in TEST_SENTENCES:
            inp = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_length=64, num_beams=4, decoder_start_token_id=model.config.decoder_start_token_id)
            print(f"   '{text}' -> '{tokenizer.decode(out[0], skip_special_tokens=True)}'")

    del model, trainer, encoder, decoder, tokenizer
    torch.cuda.empty_cache(); gc.collect()

# ------------------------------------------------------------------------
# 7. MAIN LOOP
# ------------------------------------------------------------------------
prev_info = None
for cycle in range(1, NUM_CYCLES + 1):
    for i, stage in enumerate(GENERATED_STAGES):
        should_inf = RUN_INFERENCE_EVERY_STAGE or (RUN_INFERENCE_END_OF_CYCLE and i == len(GENERATED_STAGES)-1)
        out = os.path.join(OUTPUT_ROOT, f"cycle{cycle}_{stage['name']}")
        run_stage(cycle, stage, out, prev_info, should_inf)
        prev_info = {"path": out, "enc": stage['enc'], "dec": stage['dec']}

print("\n✅ PIPELINE COMPLETE.")
