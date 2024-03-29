import json
import pickle
import nltk.stem.porter as pt
from re import template
from tqdm import tqdm
import numpy as np
import os.path as osp
import numpy as np
from rank_bm25 import BM25Okapi
import pdb

pt_stemmer = pt.PorterStemmer() 

stop_words = ["yes","no","which","this","we","what","the","can","are","likely","you","where","does",'a','he','she','is',"","an","it","some","that","there",'how','other','or',
'bu','ha','hi','wa','ga','st','am','cd','rv','hp','uk','lo','ft','dc','pm','la','th','vw','ly','ox','my','lg','dr','\"i','\'s','mm','rd','3d','ny','ma','aa','re','fo','dy','nd','a ','ii','ex',
'av','ge','dj','tp','gp','os','de','wi','un','ct','pf','ot','al','co','ye','hu','mt','sa','bp','aw','tx','ca','ne','mr','jp','cb','\'a','fe','af','ar','du','od','vy','fa','bi','ti','si','ac','pa','tw',
'nw','iv','lb','  ',' ','ep','op','te','\"e','\"a','hd','oj','rm','a\'','o\'','ba','f5','ce','yo','yo','#2','mn','og','pt','sb','ds','$1','em','sd','ho','di','pn','db','ae','4h','cv','el','rc','le','v8',
'kk','na','vh','bt','qr','om','kc','ou','ln','b5','pu','mo','\"1','ah','kg','ax','pl','li','sw','fc','jr','sk','lf','jt','7,','mu','aq','pj','ky','jc','ab','ol','1.','2.','ay','ms','4,','bc','bo','km','ty',
'll','hr','oz','fi','cm','yr','pb','su','k9','k2','sr','uv','lu','j\'','mg','jk','ri','md','â½','hs','ed','eg','fu','gb','e2','sm','jo','\'i','fm','xl','bb','5g','da','et','ro','a1','io','a2','s8','v1','vx',
'ta','ww','cy','4\'','h4','ie','ki','4e','#1','rt','eu','ag','eo','i3','o2','ea','x3','\'o','nn','u-','$2','sl','>>','ec','nj','za','ck','mc','ra','ek','$4','4o','po','kw','sq','mj','e\"','nu','xx','b6','ei',
'5%','1x','cn','\"w','m\'','i','n','t','s','o',',','m','"','&','b','w','e','c','l','y','p','-','x','d','r','v','g','k','f','#','h','u','j','/','q','!','@','(','z',':','','of','with']


def load_init():
    with open("3/train.json", "r", encoding='utf8') as ffp:
        trains = json.load(ffp)

    with open("1/valid.json", "r", encoding='utf8') as ffp:
        tests= json.load(ffp)

    with open("all_coco_dict_caption.json", "r", encoding='utf8') as ffp:
        img2caption= json.load(ffp)

    with open("v5_tripleindex_database_frequent.json", "r") as ffp:
        tripleindex_database = json.load(ffp)

    with open("v5_triplestemindex_database_frequent.json", "r") as ffp:
        triplestemindex_database = json.load(ffp)

    with open("relation2template-v2.json", "r", encoding='utf8') as ffp:
        relation2template = json.load(ffp)
    
    with open("image2text.json", "r", encoding='utf8') as ffp:
        image2text = json.load(ffp)
    return trains, tests, img2caption, tripleindex_database, triplestemindex_database, relation2template, image2text


def convert_stemkg2sentence(triplestemindex_database, relation2template):
    four_tuple = dict()
    for i,triple_stem in tqdm(enumerate(triplestemindex_database.values()), total=len(triplestemindex_database.values())):
        if triple_stem[1] in relation2template.keys():
            relation = relation2template[triple_stem[1]]
        elif triple_stem[1][-2] == "#" and triple_stem[1][-1] == "f":
            relation = "is more " + triple_stem[1][:-2] + " than"
        elif triple_stem[1][-2] == "#" and triple_stem[1][-1] == "r":
            relation = "is less " + triple_stem[1][:-2] + " than"
        else:
            relation = triple_stem[1]
        
        triple_sentence = triplestemindex_database[str(i)][0] + " " + relation + " " + triplestemindex_database[str(i)][2]
        four_tuple[i] = [triple_stem[0], triple_stem[1], triple_stem[2], triple_sentence]
    with open("four_tuple_stem.json", "w") as ffp:
        json.dump(four_tuple, ffp)
    return four_tuple


def top_500kg(trains, img2caption, image2text, four_tuple):
    okvqa_train_list = list()
    for train in tqdm(trains,total = len(trains)):
        okvqa_train_dict = dict()
        question = train["sent"]

        targets = list(train['label'].keys())
        if targets == []:
            continue
        else:
            target = targets[0]

        answer = train['label']
        
        img_id = train['img_id']
        captions = img2caption[str(train['img_id'])]
        caption_sentence = ""


        if image2text.__contains__(str(train['img_id'])):
            caption_sentence = caption_sentence + image2text[str(train['img_id'])] + " "

        for i,caption in enumerate(captions):
            cap = caption["caption"]
            if cap[-1] != ".":
                cap = cap + "."
            if i != len(captions)-1:
                caption_sentence = caption_sentence + cap + " "
            else:
                caption_sentence = caption_sentence + cap
        
        caption_sentence = caption_sentence.replace("..",".").replace(". .",".")
        sentence = question + " " + caption_sentence
        sentence = sentence.replace("?","").replace(".","").replace(",","")

        word_list = list(set(list(map(pt_stemmer.stem,sentence.split(" ")))))
        word_list_nostop = list()
        for word in word_list:
            if word not in stop_words:
                word_list_nostop.append(word)
                
        word_list_nostop = set(word_list_nostop)

        fact = dict()
        fact_500_list = list()
        for i,triple_stem in enumerate(four_tuple.values()):
            triple_stem_list = set((triple_stem[0]+ " " + triple_stem[2]).split(" "))
            if word_list_nostop & triple_stem_list:
            # if any(x in word_list_nostop for x in triple_stem_list):
                fact[triple_stem[3]] = i

        caption_word = list(set(caption_sentence.replace(".","").replace(",","").split(" ")))
        caption_del_sentence = ""
        for capp in caption_word:
            caption_del_sentence = caption_del_sentence + capp + " "
        new_sentence_forfact = question + " " +caption_del_sentence[:-1].replace("?","").replace(".","").replace(",","")
    
        tokenized_fact = [doc.split(" ") for doc in list(fact.keys())]
        bm25 = BM25Okapi(tokenized_fact)

        if len(fact)>=500:
            fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=500)
        else:
            fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=len(fact))
        for f in fact_500:
            fact_dic = dict()
            fact_dic["sentence"] = f + "."
            fact_dic["id"] = fact[f]
            fact_500_list.append(fact_dic)
        
        okvqa_train_dict["question"] = question
        okvqa_train_dict["target"] = target
        okvqa_train_dict["answer"] = answer
        okvqa_train_dict["img_id"] = img_id
        # okvqa_train_dict["score"] = score
        okvqa_train_dict["caption"] = caption_sentence
        okvqa_train_dict["fact"] = fact_500_list
        okvqa_train_list.append(okvqa_train_dict)

    print(len(okvqa_train_list))

    with open("save/vqa2_train_t5_3_v5_frequent_bm25.json", "w") as ffp:
        json.dump(okvqa_train_list, ffp)


def top_500kg_test(tests, img2caption, image2text, four_tuple):
    okvqa_test_list = list()
    print(len(tests))

    for test in tqdm(tests,total=len(tests)):
        okvqa_test_dict = dict()
        question = test["sent"]
        # 训练用的最好的答案
        targets = list(test['label'].keys())
        if targets == []:
            target = 'NNNNNNN'
            answer = {'NNNNNNN':0}
        else:
            target = targets[0]
            answer = test['label']
        # eval的时候命中任意一个即可
        
        img_id = test['img_id']
        captions = img2caption[str(test['img_id'])]
        caption_sentence = ""

        if image2text.__contains__(str(test['img_id'])):
            caption_sentence = caption_sentence + image2text[str(test['img_id'])] + " "

        for i,caption in enumerate(captions):
            cap = caption["caption"]
            if cap[-1] != ".":
                cap = cap + "."
            if i != len(captions)-1:
                caption_sentence = caption_sentence + cap + " "
            else:
                caption_sentence = caption_sentence + cap
        
        caption_sentence = caption_sentence.replace("..",".").replace(". .",".")

        sentence = question + " " + caption_sentence
        
        sentence = sentence.replace("?","").replace(".","").replace(",","")

        word_list = list(set(list(map(pt_stemmer.stem,sentence.split(" ")))))
        word_list_nostop = list()
        for word in word_list:
            if word not in stop_words:
                word_list_nostop.append(word)
        word_list_nostop = set(word_list_nostop)

        fact = dict()
        fact_500_list = list()
        for i,triple_stem in enumerate(four_tuple.values()):
            triple_stem_list = set((triple_stem[0]+ " " + triple_stem[2]).split(" "))
            if word_list_nostop & triple_stem_list:
            # if any(x in word_list_nostop for x in triple_stem_list):
                fact[triple_stem[3]] = i
        
        caption_word = list(set(caption_sentence.replace(".","").replace(",","").split(" ")))
        caption_del_sentence = ""
        for capp in caption_word:
            caption_del_sentence = caption_del_sentence + capp + " "
        new_sentence_forfact = question + " " +caption_del_sentence[:-1].replace("?","").replace(".","").replace(",","")

        tokenized_fact = [doc.split(" ") for doc in list(fact.keys())]
        bm25 = BM25Okapi(tokenized_fact)

        if len(fact)>=500:
            fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=500)
        else:
            fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=len(fact))
        for f in fact_500:
            fact_dic = dict()
            fact_dic["sentence"] = f + "."
            fact_dic["id"] = fact[f]
            fact_500_list.append(fact_dic)
        
        
        okvqa_test_dict["question"] = question
        okvqa_test_dict["target"] = target
        okvqa_test_dict["answer"] = answer
        okvqa_test_dict["img_id"] = img_id
        # okvqa_train_dict["score"] = score
        okvqa_test_dict["caption"] = caption_sentence
        okvqa_test_dict["fact"] = fact_500_list
        okvqa_test_list.append(okvqa_test_dict)

    print(len(okvqa_test_list))
    with open("save/vqa2_test111_t5_1_v5_frequent_bm25.json", "w") as ffp:
        json.dump(okvqa_test_list, ffp)


def main():
    trains, tests, img2caption, tripleindex_database, triplestemindex_database, relation2template, image2text = load_init()
    four_tuple = convert_stemkg2sentence(triplestemindex_database, relation2template)

    top_500kg(trains, img2caption, image2text, four_tuple)

    top_500kg_test(tests, img2caption, image2text, four_tuple)


if __name__ == "__main__":
    main()
