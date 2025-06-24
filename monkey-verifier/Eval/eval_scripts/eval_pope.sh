# POPE Evaluation
export HF_HOME=/shared/sheng/huggingface
export XDG_CACHE_HOME=/shared/sheng/

export CUDA_VISIBLE_DEVICES=2 

MODEL_BASE=LLaVA-RLHF-13b-v1.5-336/sft_model
MODEL_QLORA_BASE=LLaVA-RL-Fact-RLHF-13b-v1.5-336-lora-padding
MODEL_SUFFIX=$MODEL_QLORA_BASE

for POPE_CAT in popular random adversarial; do
    echo ${MODEL_SUFFIX} ${POPE_CAT}
    python model_vqa.py \
        --short_eval True \
        --model-path ./checkpoints/${MODEL_BASE}/ \
        --use-qlora True --qlora-path ./checkpoints/${MODEL_QLORA_BASE} \
        --question-file \
        ./pope/coco_pope_${POPE_CAT}.jsonl \
        --image-folder \
        ./eval_image/ \
        --answers-file \
        ./eval/pope/answer-file-${MODEL_SUFFIX}_${POPE_CAT}.jsonl --image_aspect_ratio pad --test-prompt '\nAnswer the question using a single word or phrase.'
    python summarize_eval_pope.py \
        --answers-file ./eval/pope/answer-file-${MODEL_SUFFIX}_${POPE_CAT}.jsonl \
        --label-file ./pope/coco_pope_${POPE_CAT}.jsonl
done