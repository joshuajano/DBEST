
import torch
@torch.no_grad()
def gen_init_prompt_to_emb(model, tokenizer, prompts, 
                        MAX_NUM_WORDS=77, device = 'cuda'):
    batch_size = len(prompts)
    try:
        cond_input = tokenizer(
                prompts, padding="max_length", 
                max_length=MAX_NUM_WORDS, return_tensors="pt")
        cond_emb = model.bert(cond_input.input_ids.to(device))[0]
    except:
        cond_input = tokenizer(
                prompts, padding="max_length",
                max_length=model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",)
        cond_emb = model.text_encoder(cond_input.input_ids.to(device))[0]
    
    try:
        uncond_input = tokenizer(
                        [""] * batch_size, padding="max_length", 
                        max_length=MAX_NUM_WORDS, return_tensors="pt")
        uncond_emb = model.bert(uncond_input.input_ids.to(device))[0]
    except:
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", 
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_emb = model.text_encoder(uncond_input.input_ids.to(device))[0]
    return uncond_emb, cond_emb

def gen_prompt_to_emb(model, tokenizer, prompts, 
                        MAX_NUM_WORDS=77, device = 'cuda'):
    batch_size = len(prompts)

    cond_input = tokenizer(
                prompts, padding="max_length", 
                max_length= MAX_NUM_WORDS, return_tensors="pt")
    try:
        cond_emb = model.bert(cond_input.input_ids.to(device))[0]
    except:
        cond_emb = model.text_encoder(cond_input.input_ids.to(device))[0]
    
    uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=MAX_NUM_WORDS,
            return_tensors="pt"
        )
    try:
        uncond_emb = model.bert(uncond_input.input_ids.to(device))[0]
    except:
        uncond_emb = model.text_encoder(uncond_input.input_ids.to(device))[0]
    return uncond_emb, cond_emb