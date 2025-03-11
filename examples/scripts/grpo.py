import os
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import GRPOConfig, GRPOTrainer


R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."


def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset(dataset_name, 'main')[split]

    def extract_hash_answer(text: str) -> str | None:
        try:
            return text.split("####")[1].strip()
        except IndexError:
            return None

    def process_batch(batch):
        prompts = [[
            {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "What is 2+2?"},
            {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"},
            {'role': 'user', 'content': q.strip()}
        ] for q in batch['question']]

        return {
            'prompt': prompts,
            'answer': [extract_hash_answer(a) for a in batch['answer']]
        }

    return dataset.map(process_batch, batched=True, batch_size=chunk_size)

dataset_name = 'openai/gsm8k'
dataset = preprocess_dataset(dataset_name, chunk_size=500)


def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

# reward functions
# VALID_FORMAT = re.compile(r"<reasoning>(?:(?!</?reasoning>|</?answer>).)*</reasoning>\n<answer>(?:(?!</?reasoning>|</?answer>).)*</answer>")

# def format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has the correct format."""
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [bool(VALID_FORMAT.fullmatch(r.strip())) for r in responses]
#     return [1.0 if match else 0.0 for match in matches]

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r)) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the answer is correct."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(f"Question: {prompts[0][-1]['content']}\nAnswer: {answer[0]}\nResponse: {responses[0]}\nExtracted: {extracted_responses[0]}")
    print(''.join('✅' if r == a else '❌' for r, a in zip(extracted_responses, answer)))
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# model_name = "Qwen/Qwen2.5-0.5B"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO"
run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"


# Set memory-related environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

max_prompt_length=256
max_completion_length=512

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=1e-5,
    beta=0.005, # divergence coefficient – how much the policy is allowed to deviate from the reference model. higher value – more conservative updates. Default is 0.04
    optim="adamw_8bit",
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=8,
    num_generations=4,  # group size
    gradient_accumulation_steps=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    use_vllm=True,
    # vllm_init_kwargs={
    #     "device": "cuda:0",
    #     "gpu_memory_utilization": 0.3,
    #     "max_model_len": max_prompt_length + max_completion_length,
    #     "dtype": "half",
    #     # "enable_chunked_prefill": True,
    #     # "max_num_batched_tokens": 2048,
    # },
	vllm_gpu_memory_utilization=0.3,
	vllm_dtype="half",
	vllm_max_model_len=(max_prompt_length+max_completion_length),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logit_computation_mini_batch_size=1,
    # enable_profiling=False
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2", # T4 is not supported
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=training_args.max_completion_length,
    # padding_side="left",
)
tokenizer.pad_token = tokenizer.eos_token

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

