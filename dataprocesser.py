import json
import csv
import scipy.io as scio
import pandas as pd

def process_3K():
    fr=open("AGIQA-3K/data.csv","r",encoding='utf-8')
    ls=[]
    for line in fr:
        line=line.replace("\n","")
        reader = csv.reader([line], delimiter=',')
        ls.append(next(reader))
    fr.close()
    print(ls[0])
    align = dict()
    fw=open("align.json","w",encoding='utf-8')
    for i in range(1,len(ls)):
        align[ls[i][0]] = (ls[i][7], ls[i][8])
    b = json.dumps(align,sort_keys=False,indent=4,ensure_ascii=False)
    #print(b)
    fw.write(b)
    fw.close()

def process_2023():
    prompt_flie = "AIGCIQA2023_Prompts.xlsx"
    qua_file = "AIGCIQA2023/DATA/MOS/mosz1.mat"
    qua_std = "AIGCIQA2023/DATA/STD/SD1.mat"
    aut_file = "AIGCIQA2023/DATA/MOS/mosz2.mat"
    aut_std = "AIGCIQA2023/DATA/STD/SD2.mat"
    cor_file = "AIGCIQA2023/DATA/MOS/mosz1.mat"
    cor_std = "AIGCIQA2023/DATA/STD/SD3.mat"
    qua_mean_data = scio.loadmat(qua_file)['MOSz']
    qua_std_data = scio.loadmat(qua_std)['SD']
    aut_mean_data = scio.loadmat(aut_file)['MOSz']
    aut_std_data = scio.loadmat(aut_std)['SD']
    cor_mean_data = scio.loadmat(cor_file)['MOSz']
    cor_std_data = scio.loadmat(cor_std)['SD']
    datas = [[qua_mean_data, qua_std_data, "qua"], [aut_mean_data, aut_std_data, "aut"],
              [cor_mean_data, cor_std_data, "cor"]]
    imgs = ['{}.png'.format(img) for img in range(2400)]
    for data in datas:
        fw = open("{}.json".format(data[2]),"w",encoding='utf-8')
        dic = dict()
        for i in range(2400):
            dic[imgs[i]] = (data[0][i][0], data[1][i][0])
        b = json.dumps(dic,sort_keys=False,indent=4,ensure_ascii=False)
        #print(b)
        fw.write(b)
        fw.close()
    prompts = pd.read_excel(io=prompt_flie, usecols="C").values
    dic = dict()
    for j in range(2400):
        idx = j%400//4
        print(idx)
        dic[imgs[j]] = prompts[idx][0]
    fw = open("prompts.json","w",encoding='utf-8')
    b = json.dumps(dic,sort_keys=False,indent=4,ensure_ascii=False)
    #print(b)
    fw.write(b)
    fw.close()


if __name__ == '__main__':
    process_2023()