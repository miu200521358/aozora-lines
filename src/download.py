# -*- coding: utf-8 -*-
#
import os
import urllib.request
import csv
import zipfile
import glob
import re
import time
import datetime

def download_works():

    # かぎ括弧でくくられた部分を取り出すための正規表現（最短マッチ、改行文字も文字に含む）
    lines_regex = re.compile(r"「.*?」", flags=re.DOTALL)

    # 読み仮名正規表現
    ruby_regex = re.compile(r"《.*?》", flags=re.DOTALL)

    list_file = "E:\\Development\\elasticsearch\\data\\work_list.csv"

    downloaded_data = []
    with open(list_file, encoding='utf-8', mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        downloaded_data = [row for row in reader]

        for ddata in downloaded_data:
            if not ddata["取得日"]:
                # 取得年月日がないものだけ取得
                
                # 保存ファイル名
                pathes = os.path.split(ddata["テキストファイルURL"])

                work_id = f'{int(ddata["作品ID"]):010d}'
                
                # 作業ディレクトリ
                work_dir = f'E:\\Development\\elasticsearch\\data\\works\\{work_id}'
                os.makedirs(work_dir, exist_ok=True)

                work_file = os.path.join(work_dir, pathes[-1])

                # 作業ディレクトリに保存
                urllib.request.urlretrieve(ddata["テキストファイルURL"], work_file)

                # zip解凍
                with zipfile.ZipFile(work_file) as existing_zip:
                    existing_zip.extractall(work_dir)
                    time.sleep(2)
                
                # テキストループ
                with open(os.path.join(work_dir, f'{work_id}.txt'), mode='w') as wf:
                    for txtfile in glob.glob(os.path.join(work_dir, "*.txt")):
                        with open(txtfile) as f:
                            # 全部一括で読み込む
                            s = f.read()

                            # 一致した部分をリスト形式で取り出す
                            matches = lines_regex.findall(s)

                            # 取得したセリフを全て出力
                            for match_lines in matches:
                                one_lines = re.sub(ruby_regex, '', match_lines)
                                wf.write(one_lines)
                                wf.write('\n')

                # 終わったら追加                
                ddata["取得日"] = datetime.datetime.now().isoformat()

                print(f'Finish: {ddata["作品名"]}')
                time.sleep(60)

    with open(list_file, encoding='utf-8', mode='w') as csvfile:
        writer = csv.DictWriter(csvfile, list(downloaded_data[0].keys()))
        writer.writerows(downloaded_data)

if __name__ == '__main__':
    download_works()
