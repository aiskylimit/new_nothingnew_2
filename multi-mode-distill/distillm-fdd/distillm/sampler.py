import torch
import os
from transformers import GenerationConfig


class SampleGenerator():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_new_token = self.args.max_length - self.args.max_prompt_length
        self.pad_id = tokenizer.pad_token_id
        self.generation_config = GenerationConfig(
            do_sample=args.do_sample,
            top_p=args.gen_top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            min_length=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        
    def run_sample(self, model, gen_data):
        bs = gen_data["input_ids"].size(0)
        results = {
            "input_ids": torch.ones(bs, self.args.max_length, dtype=torch.long, device=gen_data["input_ids"].device) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.args.max_length, dtype=torch.float,  device=gen_data["input_ids"].device),
            "position_ids": torch.zeros(bs, self.args.max_length, dtype=torch.long,  device=gen_data["input_ids"].device),
            "no_model_batch": torch.ones(bs, self.args.max_length, dtype=torch.long, device=gen_data["input_ids"].device) * -100,
        }
        
        model.eval()
        with torch.no_grad():
            gen_out = model.generate(
                **gen_data,
                generation_config=self.generation_config,
                max_new_tokens=self.max_new_token,
            )
            
            full_ids = gen_out.sequences
            input_ids = full_ids[:, :gen_data["input_ids"].size(1)]
            response_ids = full_ids[:, gen_data["input_ids"].size(1):]
            
            for i in range(len(input_ids)):
                input_id = input_ids[i][gen_data["attention_mask"][i].bool()]
                response_id = response_ids[i]
                eos_positions = (response_id == self.tokenizer.eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    response_id = response_id[:eos_positions[0, 0] + 1]

                # Match LMTrainDataset's causal shift: the input excludes the
                # final target, and labels start at the last prompt position.
                full_sequence = torch.cat((input_id, response_id))[:self.args.max_length]
                result_id = full_sequence[:-1]
                target_id = full_sequence[len(input_id):]

                results["input_ids"][i, :len(result_id)] = result_id
                results["attention_mask"][i, :len(result_id)] = 1
                results["position_ids"][i, :len(result_id)] = torch.arange(
                    len(result_id), device=result_id.device)
                target_start = len(input_id) - 1
                results["no_model_batch"][i, target_start:target_start + len(target_id)] = target_id
        results["attention_mask"] = results["attention_mask"].float()
        results["no_model_batch"] = results["no_model_batch"].long()
        return results
