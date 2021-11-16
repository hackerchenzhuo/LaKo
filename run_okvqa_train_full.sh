set -o errexit
gpu=1;
version="01";
iter="no";
dataset="okvqa";
model_size="large";
stream=1;
use_fact="no";
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


###########################################################
# # # # # #  attention generate # # # # # # # # # # # # # 
load_path="checkpoint/okvqa_${model_size}_batch_${batch_size}_maxLen_130/checkpoint/best_dev";
batch_size=128;
test_data="okvqa_train_t5_v5_frequent_bm25.json";
text_maxlength=130;
# -------------------
python test_reader.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --eval_data  "${test_data}" \
        --model_size "${model_size}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --text_maxlength "${text_maxlength}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --stream "${stream}" \
        --use_fact "${use_fact}" \
        --write_crossattention_scores \
        --attention_score_style "${attention_score_style}" \
        --use_last_half_layer_attention "${use_last_half_layer_attention}" \
        --version  "${version}" \
        # --write_results \
        
test_data="okvqa_test_t5_v5_frequent_bm25.json";

python test_reader.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --eval_data  "${test_data}" \
        --model_size "${model_size}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --text_maxlength "${text_maxlength}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --stream "${stream}" \
        --use_fact "${use_fact}" \
        --write_crossattention_scores \
        --attention_score_style "${attention_score_style}" \
        --use_last_half_layer_attention "${use_last_half_layer_attention}" \
        --version  "${version}" \
        # --write_results \

# ###########################################################
# ## # # # # #  train retriever # # # # # # # # # # # # #
load_path="none";
batch_size=8;
patience=5;
epoch=20;
lr=1e-4;
retriever_train_data="okvqa_train_t5_v5_frequent_bm25_${attention_part}_${model_size}_with_${attention_score_style}_${version}.json";
retriever_eval_data="okvqa_test_t5_v5_frequent_bm25_${attention_part}_${model_size}_with_${attention_score_style}_${version}.json";
fact_use_way="separate";
# -------------------

python train_retriever.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --train_data  "${retriever_train_data}" \
        --eval_data  "${retriever_eval_data}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --epochs "${epoch}" \
        --lr "${lr}" \
        --early_stop "${patience}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --use_fact "${use_fact}" \
        --fact_use_way "${fact_use_way}" \
        --version  "${version}" \
        --optim adamw \
        --scheduler linear \
        --weight_decay 1e-4 \

###########################################################
# # # # # # #  embedding generate # # # # # # # # # # # # #
load_path="checkpoint/retriever_okvqa_batch_8_content_10_from_scratch_${version}/checkpoint/best_dev";
batch_size=512;
# -------------------

python generate_fact_embeddings.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --train_data "${train_data}" \
        --eval_data "${eval_data}" \
        --per_gpu_batch_size "${batch_size}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --version  "${version}" \


###########################################################
## # # # # #  fact retriever # # # # # # # # # # # # #
batch_size=256;
facts_embeddings_path="fact_embedding_dim256_at_${version}.pkl";
fact_use_way="separate";
python fact_retrieval_small_range.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --train_data "${train_data}" \
        --eval_data "${eval_data}" \
        --per_gpu_batch_size "${batch_size}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --passages_embeddings "${facts_embeddings_path}" \
        --version  "${version}" \

###########################################################
## # # # # #  fact evaluate # # # # # # # # # # # # #
eval_data="okvqa_test_t5_v5_frequent_bm25_${version}.json";
python evaluate_retrieved_facts.py \
        --dataset "${dataset}" \
        --eval_data  "${eval_data}" \



version="v04";
iter="no";
dataset="okvqa";
model_size="base";
stream=2;
use_fact="yes";
n_context=10;
text_maxlength=130;
# mean / max / 21mean /
attention_score_style="21mean";
use_last_half_layer_attention="yes";

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


##########################################################
# # # # #  reader training # # # # # # # # # # # # # 
load_path="none";
load_path="checkpoint/vqa2.0_base_backbone/checkpoint/best_dev";
patience=5;
epoch=30;
lr=3e-5;


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


##########################################################
# # # # #  attention generate # # # # # # # # # # # # # 
load_path="checkpoint/okvqa_${model_size}_batch_${batch_size}_maxLen_130_stream_2_content_10_${iter_name}/checkpoint/best_dev";
batch_size=128;
test_data="okvqa_train_t5_v5_frequent_bm25.json";
text_maxlength=130;
# -------------------
python test_reader.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --eval_data  "${test_data}" \
        --model_size "${model_size}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --text_maxlength "${text_maxlength}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --stream "${stream}" \
        --use_fact "${use_fact}" \
        --write_crossattention_scores \
        --attention_score_style "${attention_score_style}" \
        --use_last_half_layer_attention "${use_last_half_layer_attention}" \
        --version  "${version}" \
        # --write_results \
        
test_data="okvqa_test_t5_v5_frequent_bm25.json";

python test_reader.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --eval_data  "${test_data}" \
        --model_size "${model_size}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --text_maxlength "${text_maxlength}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --stream "${stream}" \
        --use_fact "${use_fact}" \
        --write_crossattention_scores \
        --attention_score_style "${attention_score_style}" \
        --use_last_half_layer_attention "${use_last_half_layer_attention}" \
        --version  "${version}" \
        --write_results \

###########################################################
## # # # # #  train retriever # # # # # # # # # # # # #
load_path="none";
batch_size=8;
patience=5;
epoch=20;
lr=1e-4;
retriever_train_data="okvqa_train_t5_v5_frequent_bm25_${attention_part}_${model_size}_with_${attention_score_style}_${version}.json";
retriever_eval_data="okvqa_test_t5_v5_frequent_bm25_${attention_part}_${model_size}_with_${attention_score_style}_${version}.json";
fact_use_way="separate";
# -------------------

python train_retriever.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --train_data  "${retriever_train_data}" \
        --eval_data  "${retriever_eval_data}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --epochs "${epoch}" \
        --lr "${lr}" \
        --early_stop "${patience}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --use_fact "${use_fact}" \
        --fact_use_way "${fact_use_way}" \
        --version  "${version}" \
        --optim adamw \
        --scheduler linear \
        --weight_decay 1e-4 \

###########################################################
# # # # # # #  embedding generate # # # # # # # # # # # # #
load_path="checkpoint/retriever_okvqa_batch_8_content_10_from_scratch_${version}/checkpoint/best_dev";
batch_size=512;
# -------------------

python generate_fact_embeddings.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --train_data "${train_data}" \
        --eval_data "${eval_data}" \
        --per_gpu_batch_size "${batch_size}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --version  "${version}" \


###########################################################
## # # # # #  fact retriever # # # # # # # # # # # # #
batch_size=256;
facts_embeddings_path="fact_embedding_dim256_at_${version}.pkl";
fact_use_way="separate";
python fact_retrieval_small_range.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --train_data "${train_data}" \
        --eval_data "${eval_data}" \
        --per_gpu_batch_size "${batch_size}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --passages_embeddings "${facts_embeddings_path}" \
        --version  "${version}" \

###########################################################
## # # # # #  fact evaluate # # # # # # # # # # # # #
eval_data="okvqa_test_t5_v5_frequent_bm25_${version}.json";
python evaluate_retrieved_facts.py \
        --dataset "${dataset}" \
        --eval_data  "${eval_data}" \

