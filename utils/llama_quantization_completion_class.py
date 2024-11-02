from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

device = -1
if torch.cuda.is_available():
    device = 0

MAX_NEW_LENGTH = 50

MAX_LENGTH = 8192
#model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 

class LM():
    def __init__(self, model_id):
        # Quantization Configuration
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load the model with quantization
        model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tokenizer
        # Initialize the pipeline with the quantized model
        generator = pipeline(
            'text-generation', model=model_nf4, tokenizer=tokenizer, torch_dtype=torch.float16)

        self.generator = generator
        return


    # if the input is over maximum length, return True
    def check_over_length(self, prompt, report_len=False):
        input_ids = self.tokenizer(prompt)['input_ids']
        if report_len:
            print(f"length is {len(input_ids)}")
        return len(input_ids) > MAX_LENGTH-MAX_NEW_LENGTH


    def completion(self, prompt):
        with torch.no_grad():
            terminators = [
            self.tokenizer.eos_token_id,
            26, 
            2652, 
            ]
            generated_text = self.generator(prompt, do_sample=False,
                                    max_new_tokens=MAX_NEW_LENGTH,
                                    eos_token_id=terminators,
                                    num_return_sequences=1)[0]['generated_text']
        generated_text = generated_text.replace(prompt, "")
        return generated_text
