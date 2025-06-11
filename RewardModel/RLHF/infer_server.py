import json
import os
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Sequence
import einops

import torch
import transformers
from transformers import AutoTokenizer, set_seed

from lora_utils import (
    print_trainable_parameters,
    DEFAULT_PAD_TOKEN,
)
from models.reward_model import (
    RewardConfig,
    RewardModel,
)

from action_processing import ActionTokenizer

from llava import conversation as conversation_lib
from llava.model import *
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates
from llava.train.train import smart_tokenizer_and_embedding_resize

from fastapi import FastAPI, HTTPException, Request
import uvicorn
import json_numpy as json
from PIL import Image

# Set deterministic behavior
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(default=False)
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None)
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(default=500)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(default=512)
    query_len: int = field(default=None)
    response_len: int = field(default=None)
    label_names: List[str] = field(default_factory=lambda: ["index_0", "index_1", "choice"])
    padding: str = field(default="longest")
    full_finetune: bool = field(default=False)
    adam8bit: bool = field(default=False)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=4)
    lora_modules: Optional[List[str]] = field(default=None)
    lora_r: int = field(default=64)
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.0)
    report_to: str = field(default="none")
    resume_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    optim: str = field(default="paged_adamw_32bit")
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    weight_decay: float = field(default=0.0)
    learning_rate: float = field(default=0.0002)
    max_grad_norm: float = field(default=0.3)
    gradient_checkpointing: bool = field(default=True)
    do_train: bool = field(default=True)
    lr_scheduler_type: str = field(default="constant")
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=10)
    group_by_length: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=250)
    save_total_limit: int = field(default=40)
    resume_from_training: bool = field(default=False)


def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


class RobotRewardModel:
    def __init__(self):
        # Parse arguments from environment variables and defaults based on the shell script
        model_args = ModelArguments(
            model_name_or_path=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"), 
                                          "LLaVA-RLHF-7b-v1.5-224/sft_model/"),
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_vision_select_layer=-2,
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
            version="v1"
        )
        
        data_args = DataArguments(
            image_aspect_ratio='pad',
            is_multimodal=True,
            reward_prompt_file="./prompts/robot_reward_prompt.txt"
        )
        
        training_args = TrainingArguments(
            model_max_length=2048,
            query_len=1280,
            response_len=768,
            bits=16,
            lora_r=64,
            lora_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            output_dir=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"), 
                                   "LLaVA-Fact-RM-LLaVA-RLHF-7b-v1.5-224"),
            freeze_mm_mlp_adapter=True,
            group_by_length=False,
            bf16=True,
            seed=42
        )

        # Set seed for deterministic behavior
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            truncation_side="right",
            use_fast=False,
        )

        # Handle tokenizer configuration
        if model_args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=None,
                )
        elif model_args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    model_args.version
                ]
            else:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    "vicuna_v1"
                ]

        # Initialize model
        if model_args.vision_tower is not None:
            config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

            with DisableLogger():
                args = type('Args', (), {})()
                for key, value in vars(model_args).items():
                    setattr(args, key, value)
                for key, value in vars(data_args).items():
                    setattr(args, key, value)
                for key, value in vars(training_args).items():
                    setattr(args, key, value)
                
                model = RewardModel(
                    args=args,
                    config=config,
                    qlora=True,
                    checkpoint_dir=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"), "checkpoint"),
                    tokenizer=tokenizer,
                ).to(torch.bfloat16)

            model.backbone_model.config.use_cache = False  # Disable for deterministic behavior
            print_trainable_parameters(args, model)
            print("Loaded model")

            with DisableLogger():
                model_temp = model.backbone_model

            vision_tower = model_temp.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

            data_args.image_processor = vision_tower.image_processor
            model_temp.config.mm_use_im_start_end = model_args.mm_use_im_start_end

            self.tokenizer = tokenizer
            self.model = model
            self.model.eval()
            self.data_args = data_args

    def _left_pad_helper(self, ex_input_ids, batch_size):
        input_ids = [seq for seq in ex_input_ids]
        input_ids = pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=batch_size,
        )
        return input_ids

    def get_rewards(self, instruction, image_path, actions):
        batch_size = len(actions)
        action_tokenizer = ActionTokenizer(self.tokenizer)
        conv_mode = "vicuna_v1"
        conv_template = conv_templates[conv_mode].copy()

        action_in_ids = []
        instruction = instruction.lower().rstrip('.')

        for action in actions:
            action_id = np.array(action)
            if type(action_id[0]) == float:
                action_id = action_tokenizer(action)
            action_holder = ' '.join(['placeholder'] * 7)  # seven identical tokens
            
            inp = (f"shows the current observation from the robot's wrist-mounted camera. "
                   f"The robot manipulation arm is attempting to {instruction}. "
                   f"What action should the robot take to effectively accomplish the task? "
                   f"ASSISTANT: The robot should take the action: {action_holder} </s> "
                   f"USER: Please evaluate the quality of the robot action. "
                   f"A good robot action should consider different factors, "
                   f"especially interactions with surrounding objects and human preferences.\n"
                   f"ASSISTANT: Based on how humans would control the robot arm and the "
                   f"awareness of the situation, the quality score of the robot action is")
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv = conv_template.copy()
            conv.append_message(conv.roles[0], inp)
            prompt = conv.get_prompt()
            prompt = prompt.replace("<image>", " placeholder ")
            
            in_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length + 2,
                truncation=True,
            ).input_ids

            first_image_idx = (in_ids == 12983).nonzero()  # Token ID of "placeholder" is 12983
            start_idx = first_image_idx[0][1].item()
            in_ids[0, start_idx: start_idx + 1] = -200

            action_indices = (in_ids == 12983).nonzero()  # Token ID of "placeholder" is 12983
            start_idx = action_indices[0][1].item()
            in_ids[0, start_idx:start_idx + 7] = torch.tensor(action_id - 1000)

            in_ids = in_ids[:, :-1]
            in_ids = torch.tensor(in_ids, dtype=torch.long).squeeze(0)
            action_in_ids.append(in_ids)

        input_ids = self._left_pad_helper(action_in_ids, batch_size).squeeze(0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        # Process image
        processor = self.data_args.image_processor
        image = Image.open(image_path).convert("RGB")

        if self.data_args.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        images = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        model_inputs = {
            "input_ids": input_ids.cuda(0).to(torch.int64),
            "attention_mask": attention_mask.cuda(0).to(torch.int64),
            "images": images.cuda(0).to(torch.bfloat16)
        }
        
        with torch.no_grad():
            scores = self.model.forward(**model_inputs)
        
        return scores.rewards.detach().cpu().tolist()


# FastAPI application
app = FastAPI()
reward_model = None


@app.on_event("startup")
async def startup_event():
    global reward_model
    reward_model = RobotRewardModel()


@app.get("/")
async def read_root():
    return {"message": "RM server up"}


@app.post("/process")
async def process_data(request: Request):
    body = await request.body()
    data = json.loads(body)

    instruction = data.get("instruction")
    image_path = data.get("image_path")
    action = data.get("action")

    if not isinstance(instruction, str):
        raise HTTPException(status_code=400, detail="Instruction must be a string")
    if not isinstance(image_path, str):
        raise HTTPException(status_code=400, detail="Image path must be a string")

    action_array = np.array(action)

    if action_array.ndim != 2:
        raise HTTPException(status_code=400, detail="Action must be a 2D array")

    start_time = time.time()

    rewards = reward_model.get_rewards(instruction, image_path, action_array)
    
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    return {"rewards": rewards}


if __name__ == "__main__":
    # Set environment variables from shell script defaults
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "model_dir"))
    os.environ.setdefault("GPUS_PER_NODE", "1")
    
    uvicorn.run(app, host="0.0.0.0", port=3100)