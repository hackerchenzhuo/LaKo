set -o errexit
gpu=1;
version="01";
iter="no";
dataset="okvqa";
model_size="large";
stream=2;
use_fact="yes";
n_context=10;
text_maxlength=130;
# mean / max / 21mean /
attention_score_style="21mean";
use_last_half_layer_attention="no";

train_data="okvqa_train_t5_v5_frequent_bm25.json";
eval_data="okvqa_test_t5_v5_frequent_bm25.json";

if [ ${use_last_half_layer_attention} == "yes" ]
then
        attention_part="last_half_layer_attention_of";
else
        attention_part="full_attention_of";
fi
# -------------------
if [ ${model_size} == "base" ]
then
        batch_size=16;
else
        batch_size=8;
fi
# -------------------
if [ ${iter} == "yes" ]
then
        iter_name="iter_${version}";
else
        iter_name="";
fi


###########################################################
# # # # # #  reader training # # # # # # # # # # # # # 
# load_path="none";
load_path="checkpoint/vqa2.0_large_backbone/checkpoint/best_dev";
patience=3;
epoch=20;
lr=4e-5;


python train_reader.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --train_data  "${train_data}" \
        --eval_data  "${eval_data}" \
        --model_size "${model_size}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --text_maxlength "${text_maxlength}" \
        --epochs "${epoch}" \
        --lr "${lr}" \
        --early_stop "${patience}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --optim adamw \
        --scheduler linear \
        --weight_decay 1e-4 \
        --stream "${stream}" \
        --use_fact "${use_fact}" \
        --version  "${version}" \

