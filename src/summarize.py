# -*- coding: utf-8 -*-
#
import csv
import datetime
import glob
import re
import os
import pathlib
import json
from tqdm import tqdm

from sudachipy import tokenizer
from sudachipy import dictionary

LENGTH = 400
BOS = '<BOS>'
EOS = '<EOS>'

SUMMARY_LIST_FILE_PATH = 'E:\\Development\\messages\\summary_list.txt'
SUMMARY_SURFACE_DICT_FILE_PATH = 'E:\\Development\\messages\\summary_surface_dict.json'
SUMMARY_ID_DICT_FILE_PATH = 'E:\\Development\\messages\\summary_id_dict.json'

def summarize():
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C

    lines_file_regex = re.compile(r"\d+\.txt")
    ignore_texts = ["全集", "文学", "中央公論"]
    remove_texts = ["（原註）", "#", "\n", "「", "」", "　"]
    remove_text_regexs = [re.compile(r"［.*?］", flags=re.DOTALL), re.compile(r"［.*?$", flags=re.DOTALL), ]
    
    summary_text_file_path = f'E:\\Development\\messages\\summary\\summary_text_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv'
    summary_id_file_path = f'E:\\Development\\messages\\summary\\summary_id_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv'

    # サマリー用のディレクトリを作成
    sf = pathlib.Path(summary_text_file_path)
    os.makedirs(sf.parent, exist_ok=True)
    
    # これまで読み込みが終わった作品リストを取得する
    summarized_list = read_file(SUMMARY_LIST_FILE_PATH)

    # これまでの辞書を取得する
    summarized_surface_dict = read_file(SUMMARY_SURFACE_DICT_FILE_PATH)
    summarized_id_dict = read_file(SUMMARY_ID_DICT_FILE_PATH)

    wlf = open(SUMMARY_LIST_FILE_PATH, encoding='utf-8', mode='a')
    wsdf = open(SUMMARY_SURFACE_DICT_FILE_PATH, encoding='utf-8', mode='w')
    widf = open(SUMMARY_ID_DICT_FILE_PATH, encoding='utf-8', mode='w')
    wif = open(summary_id_file_path, encoding='utf-8', mode='w', newline="")
    wtf = open(summary_text_file_path, encoding='utf-8', mode='w', newline="")

    wifwriter = csv.writer(wif)
    wtfwriter = csv.writer(wtf)

    if '' not in summarized_surface_dict:
        summarized_surface_dict[''] = 0
        summarized_id_dict[0] = ''

    if BOS not in summarized_surface_dict:
        summarized_surface_dict[BOS] = 1
        summarized_id_dict[1] = BOS

    if EOS not in summarized_surface_dict:
        summarized_surface_dict[EOS] = 2
        summarized_id_dict[2] = EOS

    # 作業ディレクトリ
    for txtfile_path in tqdm(glob.glob('E:\\Development\\elasticsearch\\data\\works\\**\\*.txt')):
        if lines_file_regex.fullmatch(os.path.basename(txtfile_path)):
            if txtfile_path not in summarized_list:
                with open(txtfile_path, encoding='shift-jis', mode='r') as rf:
                    # 未サマリの台詞ファイルのみ抽出
                    for ltxt in rf.readlines():
                        is_target = True
                        for itxt in ignore_texts:
                            if itxt in ltxt:
                                is_target = False
                            
                        if is_target:
                            for rtxt in remove_texts:
                                ltxt = ltxt.replace(rtxt, "")
                            
                            for rreg in remove_text_regexs:
                                ltxt = re.sub(rreg, '', ltxt)
                            
                            for tidx, ttxt in enumerate(ltxt.split('。')):
                                if ttxt and len(ttxt) > 10:
                                    if tidx < len(ltxt.split('。')) - 1:
                                        ttxt = f'{ttxt}。'

                                    morphemes = tokenizer_obj.tokenize(ttxt, mode)
                                    tokens = [BOS] + [m.surface() for m in morphemes] + [EOS]
                                    ids = []

                                    for surface_token in tokens:
                                        if surface_token not in summarized_surface_dict:
                                            id = len(summarized_surface_dict) + 1
                                            # ID引き当て辞書と逆引き辞書を追記
                                            summarized_surface_dict[surface_token] = id
                                            summarized_id_dict[id] = surface_token
                                        else:
                                            # 登録済みの場合、IDを取得
                                            id = summarized_surface_dict[surface_token]
                                        
                                        # IDを文字列としてリストに保持
                                        ids.append(id)
                                    
                                    for i in range(LENGTH - len(tokens)):
                                        tokens.append('')
                                        ids.append(0)

                                    wifwriter.writerow(ids)
                                    wtfwriter.writerow(tokens)

                # サマリ済みリストに追記
                wlf.write(txtfile_path)
                wlf.write('\n')

    json.dump(summarized_surface_dict, wsdf, ensure_ascii=False)
    json.dump(summarized_id_dict, widf, ensure_ascii=False)

    wlf.close()
    wsdf.close()
    widf.close()
    wif.close()
    wtf.close()

def read_file(path: str):

    # ない場合、空ファイルを作成する
    if not os.path.exists(path):
        touch_file = pathlib.Path(path)
        touch_file.touch()

        if ".json" in path:
            return {}
        
    result = None
    with open(path, encoding='utf-8', mode='r') as rf:
        if ".json" in path:
            result = json.load(rf)
        else:
            result = [txt.strip() for txt in rf.readlines()]

    return result

if __name__ == '__main__':
    summarize()
