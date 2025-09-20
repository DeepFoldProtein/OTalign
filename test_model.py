import torch
from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Bo1015/proteinglm-100b-int4", trust_remote_code=True, use_fast=True)
# Use the original HuggingFace model directly
model = AutoModel.from_pretrained("Bo1015/proteinglm-100b-int4", trust_remote_code=True, torch_dtype=torch.half)
if torch.cuda.is_available():
    model = model.cuda()

# # if you don't have the single gpu with 80G memory, try the dispatch load.
# from accelerate import load_checkpoint_and_dispatch, init_empty_weights
# with init_empty_weights():
# model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
#
# model = load_checkpoint_and_dispatch(
#     model, "<your model cached dir>", device_map="auto", no_split_module_classes=["xTrimoPGLMBlock"], strict=True, dtype=dtype
# )


model.eval()

seq = "MILMCQHFSGQFSKYFLAVSSDFCHFVFPIILVSHVNFKQMKRKGFALWNDRAVPFTQGIFTTVMILLQYLHGTG"
output = tokenizer(seq, add_special_tokens=True, return_tensors="pt")
with torch.inference_mode():
    inputs = {"input_ids": output["input_ids"].cuda(), "attention_mask": output["attention_mask"].cuda()}
    output_embeddings = model(**inputs, output_hidden_states=True, return_last_hidden_state=True)  # get rid of the <eos> token

print(output_embeddings.hidden_states.shape)
print(output_embeddings)

# All tasks now use ProteinGLMModel directly
# model = ProteinGLMModel.from_pretrained("Bo1015/proteinglm-100b-int4", config=config, torch_dtype=torch.half, trust_remote_code=True)
