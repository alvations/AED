import json
import os
import torch
import torch.nn as nn
import gc
import random
import shutil
import sys
import traceback
import datetime
from datasets import load_dataset
from transformers import (
    AutoConfig,
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

# --- CONFIGURATION & ENV ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Execution Mode
RUN_ALL_COMBINATIONS = False  # Set to False to see the Random Curriculum logs

# Model Persistence & Logging Options
SAVE_MODEL = False                
SUCCESS_LOG_FILE = "success_log.tsv"
FAILURE_LOG_FILE = "failure_log.txt"

# Standard Curriculum Settings
SEED = 42
NUM_CURRICULA = 2
NUM_STAGES_PER_CYCLE = 5
EPOCHS_PER_STAGE = 3

DATA_FILE = "train.jsonl"
OUTPUT_ROOT = "./final_stable_pipeline"
PERSISTENT_ADAPTER_DIR = "./final_lora_adapter"
SAVE_TO_PERSISTENT_ADAPTER = True 
RUN_INFERENCE_EVERY_STAGE = True

ENCODER_OPTIONS = [
    {"model": "jhu-clsp/mmBERT-small",            "type": "enc-only"},
    {"model": "Qwen/Qwen2.5-0.5B",                "type": "dec-only"},
    {"model": "facebook/nllb-200-distilled-600M", "type": "seq2seq",     "use_fast": False},
    {"model": "facebook/m2m100_418M",             "type": "seq2seq",     "use_fast": False},
    {"model": "unsloth/gemma-3-270m-it",          "type": "dec-only"},
    {"model": "state-spaces/mamba-370m-hf",       "type": "state-space"}
]

DECODER_OPTIONS = [
    {"model": "jhu-clsp/mmBERT-small",             "type": "enc-only"},
    {"model": "bert-base-multilingual-cased",      "type": "enc-only"},
    {"model": "Qwen/Qwen2.5-0.5B",                 "type": "dec-only"},
    {"model": "facebook/nllb-200-distilled-600M",  "type": "seq2seq",     "use_fast": False},
    {"model": "facebook/m2m100_418M",              "type": "seq2seq",     "use_fast": False},
    {"model": "unsloth/gemma-3-270m-it",           "type": "dec-only"},
    {"model": "state-spaces/mamba-370m-hf",        "type": "state-space"}
]

if not os.path.exists(DATA_FILE):
    print(f"📝 Initializing {DATA_FILE}...")
    with open(DATA_FILE, "w") as f:
        f.write(json.dumps({"source": "Hello world", "target": "Hallo Welt", "lang_pair": "en_US-de_DE"}) + "\n")
        f.write(json.dumps({"source": "AI is evolving", "target": "KI entwickelt sich", "lang_pair": "en_US-de_DE"}) + "\n")
        f.write(json.dumps({"source": "Guten Tag", "target": "Good day", "lang_pair": "de_DE-en_US"}) + "\n")

TEST_SENTENCES = ["Hello world", "Artificial Intelligence is the future"]

# ------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'ipc_collect'): torch.cuda.ipc_collect()

def map_language_code(model_id, raw_code):
    base_lang = raw_code.split("_")[0]
    if "nllb" in model_id: return {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn"}.get(base_lang, "eng_Latn")
    elif "m2m" in model_id: return base_lang
    return None

def patch_encoder_forward(encoder_model):
    original_forward = encoder_model.forward
    def forward(*args, **kwargs):
        kwargs.pop("output_attentions", None)
        outputs = original_forward(*args, **kwargs)
        if not hasattr(outputs, "attentions"):
            try: outputs.attentions = None
            except AttributeError:
                return BaseModelOutput(last_hidden_state=outputs.last_hidden_state, hidden_states=getattr(outputs, "hidden_states", None), attentions=None)
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
        kwargs.pop("encoder_hidden_states", None); kwargs.pop("encoder_attention_mask", None); kwargs.pop("num_items_in_batch", None); kwargs.pop("output_attentions", None)
        outputs = original_forward(*args, **kwargs)
        pkv = getattr(outputs, "past_key_values", None)
        if pkv is None: pkv = getattr(outputs, "cache_params", None)
        return CausalLMOutputWithCrossAttentions(
            loss=getattr(outputs, "loss", None), logits=outputs.logits, past_key_values=pkv,
            hidden_states=getattr(outputs, "hidden_states", None), attentions=getattr(outputs, "attentions", None), cross_attentions=None
        )
    decoder_model.forward = forward
    return decoder_model

def patch_seq2seq_decoder_with_head(decoder_model, lm_head):
    original_forward = decoder_model.forward
    def forward(input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        kwargs.pop("decoder_input_ids", None); kwargs.pop("num_items_in_batch", None); kwargs.pop("output_attentions", None)
        outputs = original_forward(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, **kwargs)
        return CausalLMOutputWithCrossAttentions(
            loss=None, logits=lm_head(outputs[0]), past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions,
        )
    decoder_model.forward = forward
    return decoder_model

class RobustEncoderDecoderModel(EncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):
        if config is None: config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
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

# --- UPDATED GENERATOR WITH PRINTING ---
def generate_stages(seed, count, curriculum_num):
    random.seed(seed)
    stages = []
    print(f"\n🎲 Generating Curriculum #{curriculum_num} (Seed {seed}):")
    for i in range(1, count + 1):
        enc = random.choice(ENCODER_OPTIONS)
        dec = random.choice(DECODER_OPTIONS)
        
        lora_decision = random.choice([True, False])
        if enc.get("use_lora") or dec.get("use_lora"): lora_decision = True
        
        reset_decision = random.choice([True, False])
        
        stage = {
            "name": f"stage{i}", 
            "enc": enc["model"], "enc_type": enc["type"], "enc_use_fast": enc.get("use_fast", True),
            "dec": dec["model"], "dec_type": dec["type"], 
            "use_lora": lora_decision,
            "reset_weights": reset_decision
        }
        stages.append(stage)
        
        # --- PRINT LOG ---
        print(f"   [{stage['name']}] Enc: {stage['enc']} -> Dec: {stage['dec']} (LoRA: {stage['use_lora']} , Reset: {stage['reset_weights']})")
        
    return stages

def log_failure(stage_info, error):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    err_msg = str(error)
    tb_str = traceback.format_exc()
    print(f"❌ Stage {stage_info['name']} FAILED: {err_msg}")
    with open(FAILURE_LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\nTIMESTAMP: {timestamp}\nFAILED STAGE: {stage_info['name']}\n")
        f.write(f"Encoder: {stage_info['enc']}\nDecoder: {stage_info['dec']}\n")
        f.write(f"LoRA: {stage_info['use_lora']} | Reset: {stage_info.get('reset_weights', 'N/A')}\n")
        f.write(f"Error: {err_msg}\n{'-'*20} Stacktrace {'-'*20}\n{tb_str}\n{'='*80}\n")

# ------------------------------------------------------------------------
# RUN STAGE
# ------------------------------------------------------------------------
def run_stage(cycle_num, stage_info, output_dir, prev_lora_info, do_inference=True):
    clean_memory()
    
    enc_id, enc_type, use_fast = stage_info['enc'], stage_info['enc_type'], stage_info['enc_use_fast']
    dec_id, dec_type = stage_info['dec'], stage_info['dec_type']
    use_lora = stage_info['use_lora']
    reset_weights = stage_info.get('reset_weights', False)

    # --- DETECT BF16 SUPPORT ---
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    load_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # --- UPDATED PRINT BLOCK ---
    print(f"\n🚀 CYCLE {cycle_num} | {stage_info['name'].upper()}")
    print(f"   Encoder: {enc_id} | Decoder: {dec_id}")
    print(f"   Config: LoRA={use_lora} | ResetWeights={reset_weights}")
    print(f"   Hardware: BF16={use_bf16} -> Model Load Dtype={load_dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(enc_id, trust_remote_code=True, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def load_base_model(model_class, model_id):
        if reset_weights:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config.torch_dtype = load_dtype
            return model_class.from_config(config, trust_remote_code=True)
        else:
            return model_class.from_pretrained(model_id, trust_remote_code=True, torch_dtype=load_dtype)

    # --- ENCODER LOADING ---
    if enc_type == "seq2seq":
        full_enc = load_base_model(AutoModel, enc_id)
        encoder = full_enc.get_encoder() if hasattr(full_enc, "get_encoder") else full_enc.model.encoder
    else:
        encoder = load_base_model(AutoModel, enc_id)
        if enc_type in ["dec-only", "state-space"]: encoder = patch_encoder_forward(encoder)

    # --- DECODER LOADING ---
    decoder_already_resized = False
    if dec_type == "seq2seq":
        full_dec_model = load_base_model(AutoModelForSeq2SeqLM, dec_id)
        full_dec_model.resize_token_embeddings(len(tokenizer))
        decoder_already_resized = True
        
        if hasattr(full_dec_model, "get_decoder"): decoder = full_dec_model.get_decoder()
        elif hasattr(full_dec_model, "model") and hasattr(full_dec_model.model, "decoder"): decoder = full_dec_model.model.decoder
        else: decoder = full_dec_model 
        
        decoder = patch_seq2seq_decoder_with_head(decoder, full_dec_model.lm_head)
    elif dec_type == "enc-only":
        decoder = patch_maskedlm_forward(load_base_model(AutoModelForMaskedLM, dec_id))
    else:
        decoder = patch_causallm_forward(load_base_model(AutoModelForCausalLM, dec_id))

    if not decoder_already_resized: 
        decoder.resize_token_embeddings(len(tokenizer))
        
    model = RobustEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.to(dtype=load_dtype)

    # Global Config
    start_id = tokenizer.cls_token_id if tokenizer.cls_token_id else (tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id)
    if start_id is None: start_id = tokenizer.pad_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id
    if eos_id is None: eos_id = tokenizer.pad_token_id

    model.config.decoder_start_token_id = start_id; model.decoder.config.decoder_start_token_id = start_id
    model.config.eos_token_id = eos_id; model.decoder.config.eos_token_id = eos_id
    model.config.pad_token_id = tokenizer.pad_token_id; model.config.vocab_size = len(tokenizer)
    model.config.num_beams = 4
    model.config.gradient_checkpointing = True

    if use_lora:
        print("   Applying PEFT (LoRA)...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64, lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense", "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj", "x_proj", "dt_proj", "out_proj"],
            modules_to_save=["embed_tokens", "lm_head"] 
        )
        model = get_peft_model(model, peft_config)
        if prev_lora_info:
            sf_file = os.path.join(prev_lora_info['path'], "adapter_model.safetensors")
            bin_file = os.path.join(prev_lora_info['path'], "adapter_model.bin")
            state_dict = load_file(sf_file) if os.path.exists(sf_file) else (torch.load(bin_file, map_location="cpu") if os.path.exists(bin_file) else None)
            if state_dict: model.load_state_dict(state_dict, strict=False)
    else:
        print("   ⚠️ Full finetuning enabled.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Data
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

    # Train
    trainer = Seq2SeqTrainer(
        model=model, train_dataset=tokenized, tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        args=Seq2SeqTrainingArguments(
            output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=16, 
            learning_rate=2e-4, num_train_epochs=EPOCHS_PER_STAGE, 
            bf16=use_bf16, fp16=use_fp16, 
            save_strategy="no", report_to="none"
        )
    )
    trainer.train()
    
    print(f"   ✅ Training complete. Logging to {SUCCESS_LOG_FILE}...")
    if not os.path.exists(SUCCESS_LOG_FILE):
        with open(SUCCESS_LOG_FILE, "w") as f: f.write("Encoder\tDecoder\tLoRA\tReset_Weights\n")
    with open(SUCCESS_LOG_FILE, "a") as f: f.write(f"{enc_id}\t{dec_id}\t{use_lora}\t{reset_weights}\n")

    if SAVE_MODEL:
        print(f"   💾 Saving model to {output_dir}...")
        try: model.save_pretrained(output_dir, safe_serialization=False)
        except Exception as e: torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    else:
        print("   🚫 Model saving disabled (SAVE_MODEL=False).")

    # Inference
    if do_inference:
        inference_beams = 1 if dec_type == "state-space" else 4
        print(f"\n🧪 INFERENCE TEST (Beams={inference_beams}):"); 
        
        model.eval()
        model.config.use_cache = False 
        
        inference_model = model
        for text in TEST_SENTENCES:
            inp = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = inference_model.generate(
                    **inp, max_length=64, 
                    num_beams=inference_beams, 
                    decoder_start_token_id=start_id, eos_token_id=eos_id, pad_token_id=tokenizer.pad_token_id,
                    use_cache=False
                )
            print(f"   '{text}' -> '{tokenizer.decode(out[0], skip_special_tokens=True)}'")
        
        del inference_model

    # Cleanup
    del trainer
    model.cpu() 
    del model, encoder, decoder, tokenizer
    clean_memory()
    print("   🧹 Memory Cleaned.")

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
if __name__ == '__main__':
    last_lora_info = None

    if RUN_ALL_COMBINATIONS:
        print(f"\n⚡ RUN_ALL_COMBINATIONS=True: Grid Search")
        print(f"   Dimensions: Encoder x Decoder x LoRA x ResetWeights")
        grid_stages = []
        id_counter = 1
        for enc in ENCODER_OPTIONS:
            for dec in DECODER_OPTIONS:
                for use_lora in [False, True]:
                    for reset_weights in [False, True]:
                        enc_name = enc['model'].split('/')[-1]; dec_name = dec['model'].split('/')[-1]
                        stage_name = f"grid_{id_counter:03d}_{enc_name}_to_{dec_name}_{'LoRA' if use_lora else 'Full'}_{'Reset' if reset_weights else 'Pretrained'}"
                        grid_stages.append({
                            "name": stage_name,
                            "enc": enc["model"], "enc_type": enc["type"], "enc_use_fast": enc.get("use_fast", True),
                            "dec": dec["model"], "dec_type": dec["type"], 
                            "use_lora": use_lora, "reset_weights": reset_weights
                        })
                        id_counter += 1
        print(f"📋 Generated {len(grid_stages)} combinations.")
        
        for i, stage in enumerate(grid_stages):
            out_dir = os.path.join(OUTPUT_ROOT, stage['name'])
            try: run_stage(1, stage, out_dir, prev_lora_info=None, do_inference=RUN_INFERENCE_EVERY_STAGE)
            except Exception as e: log_failure(stage, e)

    else:
        print(f"\n🚀 STARTING META-PIPELINE: {NUM_CURRICULA} Random Curricula")
        for curriculum_num in range(1, NUM_CURRICULA + 1):
            GENERATED_STAGES = generate_stages(SEED + curriculum_num - 1, NUM_STAGES_PER_CYCLE, curriculum_num)
            
            if curriculum_num > 1 and os.path.exists(PERSISTENT_ADAPTER_DIR):
                info_file = os.path.join(PERSISTENT_ADAPTER_DIR, "info.json")
                if os.path.exists(info_file):
                    with open(info_file, "r") as f: last_lora_info = json.load(f)
                else: last_lora_info = None
            else: last_lora_info = None

            for i, stage in enumerate(GENERATED_STAGES):
                out_dir = os.path.join(OUTPUT_ROOT, f"curriculum{curriculum_num}_cycle1_{stage['name']}")
                try:
                    run_stage(1, stage, out_dir, last_lora_info, RUN_INFERENCE_EVERY_STAGE or (i == len(GENERATED_STAGES)-1))
                    if stage['use_lora']: last_lora_info = {"path": out_dir, "enc": stage['enc'], "dec": stage['dec']}
                except Exception as e: log_failure(stage, e)

            if SAVE_TO_PERSISTENT_ADAPTER and last_lora_info:
                if os.path.exists(PERSISTENT_ADAPTER_DIR): shutil.rmtree(PERSISTENT_ADAPTER_DIR)
                shutil.copytree(last_lora_info['path'], PERSISTENT_ADAPTER_DIR)
                with open(os.path.join(PERSISTENT_ADAPTER_DIR, "info.json"), "w") as f:
                    json.dump({"enc": last_lora_info["enc"], "dec": last_lora_info["dec"], "path": PERSISTENT_ADAPTER_DIR}, f, indent=4)
        print("\n✅ ALL CURRICULA COMPLETE.")
