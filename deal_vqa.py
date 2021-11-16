import json
import os.path as osp
dataset = "vqa2.0"
eval_data = "vqa2.0_test_t5_v5_frequent_bm25.json"
output_data_10000 = "vqa2.0_test_t5_v5_frequent_bm25_top_10000.json"
output_data_5000 = "vqa2.0_test_t5_v5_frequent_bm25_top_5000.json"


this_dir = osp.dirname(__file__)
data_path = osp.abspath(osp.join(this_dir, '..', '..', 'data', 'LaKo'))
cache_dir = osp.abspath(osp.join(this_dir, '..', '..', 'data', '.cache', 'transformers'))
eval_data_path = osp.join(data_path, dataset, eval_data)
with open(eval_data_path, 'r') as fin:
    eval_examples = json.load(fin)


print("save...")
eval_examples_10000 = eval_examples[:10000]
del eval_examples
eval_examples_5000 = eval_examples_10000[:5000]

output_train_data_path = osp.join(data_path, dataset, output_data_10000)
with open(output_train_data_path, 'w') as fw:
    json.dump(eval_examples_10000, fw)
output_train_data_path = osp.join(data_path, dataset, output_data_5000)
with open(output_train_data_path, 'w') as fw:
    json.dump(eval_examples_5000, fw)
print("done..")
