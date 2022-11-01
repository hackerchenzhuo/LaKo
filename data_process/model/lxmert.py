from transformers import LxmertForQuestionAnswering, LxmertTokenizer, LxmertConfig
import torch.nn as nn
import pdb


class LXMERT(nn.Module):
    def __init__(self, args, num_labels=None):
        super(LXMERT, self).__init__()
        self.args = args
        cache_dir = '/data/chenzhuo/data/.cache/transformers'
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir=cache_dir)
        if len(args.load) <= 1 and not args.split_segment:
            self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased", cache_dir=cache_dir)
        else:
            if args.split_segment:
                # embedding 修改之后finetune无意义
                assert not args.finetune
                config_class = LxmertConfig(type_vocab_size=2)  # 2 is enough ? 只决定神经网络type_vocab的size
            else:
                config_class = LxmertConfig()
            # pdb.set_trace()
            self.lxmert_vqa = LxmertForQuestionAnswering(config_class)
        # pdb.set_trace()
        if num_labels is not None and self.lxmert_vqa.num_qa_labels != num_labels:
            print(f"number of answer candidate labels change: {num_labels}")
            self.lxmert_vqa.resize_num_qa_labels(num_labels)

        # self.lxmert_vqa.eval()

    def forward(self, feats, boxes, sent, fact=None):

        ########
        # pdb.set_trace()
        test_question = sent
        if fact is None:
            inputs = self.lxmert_tokenizer(
                test_question,
                padding="max_length",
                max_length=50,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

        else:
            inputs = self.lxmert_tokenizer(
                fact, test_question,
                padding="max_length",
                max_length=50,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

        inputs.attention_mask = inputs.attention_mask.cuda()
        inputs.input_ids = inputs.input_ids.cuda()
        inputs.token_type_ids = inputs.token_type_ids.cuda()

        output_attention = self.args.output_attention

        output_vqa = self.lxmert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=feats,  # features,
            visual_pos=boxes,  # normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            output_attentions=output_attention,  # attention
        )

        # pdb.set_trace()
        logit = output_vqa["question_answering_score"]

        if output_attention:
            language_attentions = output_vqa["language_attentions"]
            vision_attentions = output_vqa["vision_attentions"]
            cross_encoder_attentions = output_vqa["cross_encoder_attentions"]
            return logit, language_attentions, vision_attentions, cross_encoder_attentions, inputs.input_ids
        else:
            return logit
        # label = output_vqa["question_answering_score"].argmax(-1) # pred_vqa
        ########


# class LXMERT(nn.Module):
#     def __init__(self, use_lrp=False):
#         self.vqa_answers = utils.get_data(VQA_URL)

#         # load models and model components
#         self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
#         self.frcnn_cfg.MODEL.DEVICE = "cuda"

#         self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

#         self.image_preprocess = Preprocess(self.frcnn_cfg)

#         self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

#         if use_lrp:
#             self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")
#         else:
#             self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")

#         self.lxmert_vqa.eval()
#         self.model = self.lxmert_vqa

    # def forward(self, item):
    #     URL, question = item

    #     self.image_file_path = URL

    #     # run frcnn
    #     images, sizes, scales_yx = self.image_preprocess(URL)
    #     output_dict = self.frcnn(
    #         images,
    #         sizes,
    #         scales_yx=scales_yx,
    #         padding="max_detections",
    #         max_detections= self.frcnn_cfg.max_detections,
    #         return_tensors="pt"
    #     )
    #     inputs = self.lxmert_tokenizer(
    #         question,
    #         truncation=True,
    #         return_token_type_ids=True,
    #         return_attention_mask=True,
    #         add_special_tokens=True,
    #         return_tensors="pt"
    #     )
    #     self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
    #     self.text_len = len(self.question_tokens)
    #     # Very important that the boxes are normalized
    #     normalized_boxes = output_dict.get("normalized_boxes")
    #     features = output_dict.get("roi_features")
    #     self.image_boxes_len = features.shape[1]
    #     self.bboxes = output_dict.get("boxes")
    #     self.output = self.lxmert_vqa(
    #         input_ids=inputs.input_ids.to("cuda"),
    #         attention_mask=inputs.attention_mask.to("cuda"),
    #         visual_feats=features.to("cuda"),
    #         visual_pos=normalized_boxes.to("cuda"),
    #         token_type_ids=inputs.token_type_ids.to("cuda"),
    #         return_dict=True,
    #         output_attentions=False,
    #     )
    #     return self.output
