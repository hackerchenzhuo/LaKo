gpu=1;
# load_path="none";
load_path="checkpoint/okvqa_large_batch_8_maxLen_130_stream_2_content_10_/checkpoint/best_dev";
# -------------------
dataset="okvqa";
batch_size=128;
stream=2;
# -------------------
model_size="large";

eval_data="okvqa_test_t5_v5_frequent_bm25.json";
# mean / max / 21mean /
attention_score_style="21mean";
consider_context_attention="no";
use_last_half_layer_attention="yes";
n_context=10;
text_maxlength=130;
use_fact="yes";
# -------------------
python test_reader.py \
        --gpu "${gpu}" \
        --dataset "${dataset}" \
        --eval_data  "${eval_data}" \
        --model_size "${model_size}" \
        --per_gpu_batch_size "${batch_size}" \
        --n_context "${n_context}" \
        --text_maxlength "${text_maxlength}" \
        --model_path "${load_path}" \
        --checkpoint_dir checkpoint \
        --stream "${stream}" \
        --use_fact "${use_fact}" \
        --write_results \
        # --write_crossattention_scores \
        # --attention_score_style "${attention_score_style}" \
        # --use_last_half_layer_attention "${use_last_half_layer_attention}" \
        # --consider_context_attention "${consider_context_attention}" \
        
        




