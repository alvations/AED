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
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderConfig
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

# --- FORCE SINGLE-GPU OPERATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------
SEED = 42
NUM_CURRICULA = 2
NUM_STAGES_PER_CYCLE = 5
EPOCHS_PER_STAGE = 3

DATA_FILE = "train.jsonl"
OUTPUT_ROOT = "./final_stable_pipeline"
PERSISTENT_ADAPTER_DIR = "./final_lora_adapter"

# Persistence Controls
LOAD_FROM_PERSISTENT_ADAPTER = False
SAVE_TO_PERSISTENT_ADAPTER = True

# Inference Controls
RUN_INFERENCE_EVERY_STAGE = False
RUN_INFERENCE_END_OF_CYCLE = True

# --- MODEL OPTIONS ---
ENCODER_OPTIONS = [
    {"model": "jhu-clsp/mmBERT-small",            "type": "enc-only"},
    #{"model": "tencent/HY-MT1.5-1.8B",            "type": "dec-only",    "use_lora": True}, 
    #{"model": "Qwen/Qwen2.5-0.5B",                "type": "dec-only"},
    {"model": "facebook/nllb-200-distilled-600M", "type": "seq2seq",     "use_fast": False},
    {"model": "facebook/m2m100_418M",             "type": "seq2seq",     "use_fast": False},
    {"model": "unsloth/gemma-3-270m-it",          "type": "dec-only"},
    {"model": "state-spaces/mamba-370m-hf",       "type": "state-space"}
]

DECODER_OPTIONS = [
    {"model": "jhu-clsp/mmBERT-small",             "type": "enc-only"},
    {"model": "bert-base-multilingual-cased",      "type": "enc-only"},
    #{"model": "tencent/HY-MT1.5-1.8B",             "type": "dec-only",    "use_lora": True},
    #{"model": "Qwen/Qwen2.5-0.5B",                 "type": "dec-only"},
    {"model": "facebook/nllb-200-distilled-600M",  "type": "seq2seq",     "use_fast": False},
    {"model": "facebook/m2m100_418M",              "type": "seq2seq",     "use_fast": False},
    {"model": "unsloth/gemma-3-270m-it",           "type": "dec-only"},
    {"model": "state-spaces/mamba-370m-hf",        "type": "state-space"}
]

# Create Dummy Data
if not os.path.exists(DATA_FILE):
    print(f"📝 Initializing {DATA_FILE}...")
    with open(DATA_FILE, "w") as f:
        f.write(json.dumps({"source": "Hello world", "target": "Hallo Welt", "lang_pair": "en_US-de_DE"}) + "\n")
        f.write(json.dumps({"source": "AI is evolving", "target": "KI entwickelt sich", "lang_pair": "en_US-de_DE"}) + "\n")
        f.write(json.dumps({"source": "Guten Tag", "target": "Good day", "lang_pair": "de_DE-en_US"}) + "\n")

TEST_SENTENCES = ["Hello world", "Artificial Intelligence is the future"]

# ------------------------------------------------------------------------
# 2. HELPER: DYNAMIC LANGUAGE MAPPING
# ------------------------------------------------------------------------
def map_language_code(model_id, raw_code):
    base_lang = raw_code.split("_")[0]
    if "nllb" in model_id:
        mapping = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn"}
        return mapping.get(base_lang, "eng_Latn")
    elif "m2m" in model_id:
        return base_lang
    return None

# ------------------------------------------------------------------------
# 3. HELPER: COMPATIBILITY PATCHES
# ------------------------------------------------------------------------
def patch_encoder_forward(encoder_model):
    original_forward = encoder_model.forward
    def forward(*args, **kwargs):
        # Mamba and some others don't support output_attentions
        kwargs.pop("output_attentions", None)
        
        outputs = original_forward(*args, **kwargs)
        
        # FIX: Ensure 'attentions' attribute exists for EncoderDecoderModel compatibility
        if not hasattr(outputs, "attentions"):
            try:
                outputs.attentions = None
            except AttributeError:
                # If output is a frozen dataclass, return a compatible wrapper
                return BaseModelOutput(
                    last_hidden_state=outputs.last_hidden_state,
                    hidden_states=getattr(outputs, "hidden_states", None),
                    attentions=None
                )
        return outputs
        
    encoder_model.forward = forward
    return encoder_model

def patch_maskedlm_forward(decoder_model):
    original_forward = decoder_model.forward
    def forward(*args, **kwargs):
        kwargs.pop("use_cache", None); kwargs.pop("past_key_values", None); kwargs.pop("num_items_in_batch", None); kwargs.pop("output_attentions", None)
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
        # NOTE: Removing encoder_hidden_states here effectively disables cross-attention 
        # for standard CausalLMs (like Qwen/Gemma/Mamba) that don't support it natively.
        kwargs.pop("encoder_hidden_states", None); kwargs.pop("encoder_attention_mask", None); 
        kwargs.pop("num_items_in_batch", None); kwargs.pop("output_attentions", None)
        
        outputs = original_forward(*args, **kwargs)
        
        # Robustly handle missing past_key_values (e.g. for Mamba which uses cache_params)
        pkv = getattr(outputs, "past_key_values", None)
        if pkv is None:
             pkv = getattr(outputs, "cache_params", None)

        return CausalLMOutputWithCrossAttentions(
            loss=getattr(outputs, "loss", None), 
            logits=outputs.logits, 
            past_key_values=pkv,
            hidden_states=getattr(outputs, "hidden_states", None), 
            attentions=getattr(outputs, "attentions", None),
            cross_attentions=None
        )
    decoder_model.forward = forward
    return decoder_model

def patch_seq2seq_decoder_with_head(decoder_model, lm_head):
    """
    Patches a raw seq2seq decoder (which outputs hidden states) to apply the lm_head 
    and output CausalLMOutputWithCrossAttentions (which contains logits).
    """
    original_forward = decoder_model.forward
    
    def forward(input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        # Clean kwargs that raw decoder doesn't expect or that cause issues
        kwargs.pop("decoder_input_ids", None)
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("output_attentions", None)

        # Forward pass through the decoder (transformer)
        outputs = original_forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            encoder_hidden_states=encoder_hidden_states, 
            encoder_attention_mask=encoder_attention_mask, 
            **kwargs
        )
        
        # outputs[0] is last_hidden_state
        hidden_states = outputs[0]
        
        # Project hidden states to vocabulary using the provided head
        logits = lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            loss=None, # Loss is computed by EncoderDecoderModel
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        
    decoder_model.forward = forward
    return decoder_model

# ------------------------------------------------------------------------
# 4. ROBUST MODEL CLASS
# ------------------------------------------------------------------------
class RobustEncoderDecoderModel(EncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):
        if config is None: config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        super(EncoderDecoderModel, self).__init__(config)
        self.encoder, self.decoder = encoder, decoder
        self.decoder.config.is_decoder = True
        self.decoder.config.add_cross_attention = True
        
        # Initialize projection if hidden sizes differ
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
            init_std = getattr(self.decoder.config, "initializer_range", 0.02)
            self.enc_to_dec_proj.weight.data.normal_(mean=0.0, std=init_std)
            if self.enc_to_dec_proj.bias is not None: self.enc_to_dec_proj.bias.data.zero_()
        else:
            self.enc_to_dec_proj = None

# ------------------------------------------------------------------------
# 5. CURRICULUM GENERATION
# ------------------------------------------------------------------------
def generate_stages(seed, count, curriculum_num):
    random.seed(seed)
    stages = []
    print(f"\n🎲 Generating Curriculum #{curriculum_num} (Seed {seed}):")
    for i in range(1, count + 1):
        enc = random.choice(ENCODER_OPTIONS)
        dec = random.choice(DECODER_OPTIONS)
        lora_decision = random.choice([True, False])
        if enc.get("use_lora") or dec.get("use_lora"):
            lora_decision = True
        stage = {
            "name": f"stage{i}", "enc": enc["model"], "enc_type": enc["type"],
            "enc_use_fast": enc.get("use_fast", True),
            "dec": dec["model"], "dec_type": dec["type"],
            "use_lora": lora_decision
        }
        stages.append(stage)
        print(f"   [{stage['name']}] Enc: {stage['enc']} -> Dec: {stage['dec']} (LoRA: {stage['use_lora']})")
    return stages

# ------------------------------------------------------------------------
# 6. SMART WEIGHT LOADER
# ------------------------------------------------------------------------
def smart_load_weights(model, adapter_path, load_enc, load_dec, load_embed):
    print(f"♻️  Loading LoRA weights from {adapter_path}...")
    sf_file = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_file = os.path.join(adapter_path, "adapter_model.bin")
    state_dict = load_file(sf_file) if os.path.exists(sf_file) else (torch.load(bin_file, map_location="cpu") if os.path.exists(bin_file) else None)
    
    if not state_dict:
        print("   ⚠️ Source adapter file not found. Starting Fresh.")
        return
    filtered = {k: v for k, v in state_dict.items() if (("embed_tokens" in k or "lm_head" in k) and load_embed) or ("encoder" in k and load_enc) or ("decoder" in k and load_dec)}
    if filtered: 
        model.load_state_dict(filtered, strict=False)
        print(f"   ✅ Transferred {len(filtered)} compatible LoRA tensors.")
    else:
        print("   ⚠️ No compatible LoRA tensors found. Initializing fresh adapters.")

# ------------------------------------------------------------------------
# 7. TRAINING ENGINE
# ------------------------------------------------------------------------
def run_stage(cycle_num, stage_info, output_dir, prev_lora_info, do_inference=True):
    # Aggressive cleanup
    gc.collect(); torch.cuda.empty_cache()

    enc_id, enc_type, use_fast = stage_info['enc'], stage_info['enc_type'], stage_info['enc_use_fast']
    dec_id, dec_type = stage_info['dec'], stage_info['dec_type']
    use_lora = stage_info['use_lora']
    
    print(f"\n🚀 CYCLE {cycle_num} | {stage_info['name'].upper()} (LoRA: {use_lora})")
    print(f"   Tokenizer: {enc_id} (use_fast={use_fast})")

    # --- TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(enc_id, trust_remote_code=True, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # --- MODELS (Load to CPU first) ---
    print("   Loading Base Models to CPU...")
    
    # 1. Load Encoder
    if enc_type == "seq2seq":
        full_enc = AutoModel.from_pretrained(enc_id, trust_remote_code=True)
        encoder = full_enc.get_encoder() if hasattr(full_enc, "get_encoder") else full_enc.model.encoder
    else:
        encoder = AutoModel.from_pretrained(enc_id, trust_remote_code=True)
        if enc_type in ["dec-only", "state-space"]:
             encoder = patch_encoder_forward(encoder)

    # 2. Load Decoder
    decoder_already_resized = False

    if dec_type == "seq2seq":
        # Load the FULL model first
        full_dec_model = AutoModelForSeq2SeqLM.from_pretrained(dec_id, trust_remote_code=True)
        
        # Resize embeddings on the FULL model NOW so head and decoder remain synced
        print(f"   Resizing Seq2Seq embeddings to {len(tokenizer)}...")
        full_dec_model.resize_token_embeddings(len(tokenizer))
        decoder_already_resized = True
        
        # Extract Decoder and Head
        if hasattr(full_dec_model, "get_decoder"):
            decoder = full_dec_model.get_decoder()
        elif hasattr(full_dec_model, "model") and hasattr(full_dec_model.model, "decoder"):
            decoder = full_dec_model.model.decoder
        else:
            decoder = full_dec_model # Fallback
            
        lm_head = full_dec_model.lm_head
        
        # Patch the decoder to use the head and output logits
        decoder = patch_seq2seq_decoder_with_head(decoder, lm_head)
        
    elif dec_type == "enc-only":
        decoder = patch_maskedlm_forward(AutoModelForMaskedLM.from_pretrained(dec_id, trust_remote_code=True))
    else:
        # Handles 'dec-only' and 'state-space' (Mamba)
        decoder = patch_causallm_forward(AutoModelForCausalLM.from_pretrained(dec_id, trust_remote_code=True))

    # Resize if we haven't done it yet (for non-seq2seq models)
    if not decoder_already_resized:
        decoder.resize_token_embeddings(len(tokenizer))
        
    model = RobustEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # --- CONFIG ---
    start_id = tokenizer.cls_token_id if tokenizer.cls_token_id else (tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id)
    if start_id is None: start_id = tokenizer.pad_token_id
    
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id
    if eos_id is None: eos_id = tokenizer.pad_token_id

    model.config.decoder_start_token_id = start_id
    model.decoder.config.decoder_start_token_id = start_id
    model.config.eos_token_id = eos_id
    model.decoder.config.eos_token_id = eos_id
    
    model.config.pad_token_id, model.config.vocab_size = tokenizer.pad_token_id, len(tokenizer)
    model.config.max_length, model.config.num_beams, model.config.use_cache = 128, 4, False
    model.config.gradient_checkpointing = True

    # --- PEFT ---
    if use_lora:
        print("   Applying PEFT (LoRA)...")
        target_modules = [
            "query", "key", "value", "dense", "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj", "in_proj", "x_proj", "dt_proj", "out_proj"
        ]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64, lora_dropout=0.1,
            target_modules=target_modules,
            modules_to_save=["embed_tokens", "lm_head"] 
        )
        model = get_peft_model(model, peft_config)

        if prev_lora_info:
            smart_load_weights(model, prev_lora_info['path'], (prev_lora_info['enc']==enc_id), (prev_lora_info['dec']==dec_id), (prev_lora_info['enc']==enc_id))
        
        model.print_trainable_parameters()
    else:
        print("   ⚠️ Full finetuning enabled (LoRA skipped).")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
        args=Seq2SeqTrainingArguments(
            output_dir=output_dir, per_device_train_batch_size=1, 
            gradient_accumulation_steps=16, learning_rate=2e-4, 
            num_train_epochs=EPOCHS_PER_STAGE, 
            fp16=True if torch.cuda.is_available() else False, 
            save_strategy="no", report_to="none"
        )
    )
    trainer.train()
    
    # Save model or adapter
    model.save_pretrained(output_dir)
    
    if do_inference:
        print(f"\n🧪 INFERENCE TEST:"); model.eval(); model.config.use_cache = True
        inference_model = model
        
        for text in TEST_SENTENCES:
            inp = tokenizer(text, return_tensors="pt").to(inference_model.device)
            with torch.no_grad():
                out = inference_model.generate(
                    **inp, max_length=64, num_beams=4, 
                    decoder_start_token_id=inference_model.config.decoder_start_token_id,
                    output_attentions=False,
                    eos_token_id=inference_model.config.eos_token_id,
                    pad_token_id=inference_model.config.pad_token_id
                )
            print(f"   '{text}' -> '{tokenizer.decode(out[0], skip_special_tokens=True)}'")

    del model, trainer, encoder, decoder, tokenizer
    gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------------------------------
# 8. MAIN LOOP
# ------------------------------------------------------------------------
if __name__ == '__main__':
    last_lora_info = None

    print(f"\n🚀 STARTING META-PIPELINE: {NUM_CURRICULA} Curricula")

    for curriculum_num in range(1, NUM_CURRICULA + 1):
        print(f"\n{'='*70}")
        print(f"  RUNNING CURRICULUM #{curriculum_num} / {NUM_CURRICULA}")
        print(f"{'='*70}")

        GENERATED_STAGES = generate_stages(SEED + curriculum_num - 1, NUM_STAGES_PER_CYCLE, curriculum_num)
        
        if curriculum_num > 1 and os.path.exists(PERSISTENT_ADAPTER_DIR):
            print(f"Attempting to load persistent adapter from {PERSISTENT_ADAPTER_DIR}")
            info_file = os.path.join(PERSISTENT_ADAPTER_DIR, "info.json")
            if os.path.exists(info_file):
                with open(info_file, "r") as f:
                    last_lora_info = json.load(f)
                print(f"   Successfully loaded info: Enc={last_lora_info['enc']}, Dec={last_lora_info['dec']}")
            else:
                last_lora_info = None
        else:
             last_lora_info = None

        for i, stage in enumerate(GENERATED_STAGES):
            should_inf = RUN_INFERENCE_EVERY_STAGE or (RUN_INFERENCE_END_OF_CYCLE and i == len(GENERATED_STAGES)-1)
            out_dir = os.path.join(OUTPUT_ROOT, f"curriculum{curriculum_num}_cycle1_{stage['name']}")
            
            run_stage(1, stage, out_dir, last_lora_info, should_inf)
            
            if stage['use_lora']:
                last_lora_info = {"path": out_dir, "enc": stage['enc'], "dec": stage['dec']}

        if SAVE_TO_PERSISTENT_ADAPTER and last_lora_info:
            print(f"\n✅ Saving final adapter from {last_lora_info['path']} to {PERSISTENT_ADAPTER_DIR}")
            if os.path.exists(PERSISTENT_ADAPTER_DIR):
                shutil.rmtree(PERSISTENT_ADAPTER_DIR)
            shutil.copytree(last_lora_info['path'], PERSISTENT_ADAPTER_DIR)
            
            info_to_save = {
                "enc": last_lora_info["enc"],
                "dec": last_lora_info["dec"],
                "path": PERSISTENT_ADAPTER_DIR
            }
            with open(os.path.join(PERSISTENT_ADAPTER_DIR, "info.json"), "w") as f:
                json.dump(info_to_save, f, indent=4)
    
    print("\n✅ ALL CURRICULA COMPLETE.")
