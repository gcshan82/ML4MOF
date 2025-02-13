import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from collections import Counter
from mordred import Calculator, descriptors
import shap
from tqdm import tqdm

input_file = 'hMOF_CO2_2.5_small_mofid.csv'
output_file = 'output_large.csv'
data = pd.read_csv(input_file)
#创建字典count_dict，统计各种元素和信息
count_dict = {'cl': [], 'o': [], 'n': [], 'br': [], 'c': [], 'f': [], 'c1ccncc1': [], '[nh2]': [], '[nh]': [], '[oh]': [], 'structure': [], 'cat': [], 'metal': [], 'label': []}
#创建字典linker_dict，存储linker,metal和label相关信息
linker_dict = {'linker': [], 'metal': [], 'label': []}
str_list = []
i = 0
# 使用split函数按照'.'进行分割
# 有机配体元素统计
for index, row in data.iterrows():              #遍历data DataFrame的每一行，获取索引和行内容
    smiles = row['smiles']                      #获取当前行中'smiles'列的值
    label = row['label']
    count_dict['label'].append(label)           #将标签添加到count_dict中的'label'列表中
    separator = '&&'                            #定义分隔符为'&&'
    smiles_ele = smiles.split(separator, 1)[0]  #通过分隔符提取前半部分(金属位点加有机配体部分)
    smiles_str = smiles.split(separator, 1)[1]  #通过分隔符提取后半部分(提取拓扑结构与cat部分)
    smiles_gnt = smiles_ele
    smiles_metal = smiles_ele
    linker_list = []
    new_list = []

    linker_dict['label'].append(label)             #将标签添加到linker_dict中的'label'列表中

    split_parts = smiles_ele.split('.')            #按照'.'字符对smiles_ele进行切分，得到包含各个部分的列表(包含金属团簇和有机配体)
    mof_metal_elements = ['Zn', 'Cu', 'V', 'Zr']            #定义MOF金属元素列表
    mof_ori_elements = ['cl', 'o', 'n', 'br', 'c', 'f']     #定义MOF有机元素列表


    for split_part in split_parts:                 #遍历分割后的每个部分
        # 初始化一个计数器，用于记录包含特定字符的数量
        contains_target_char = False               #初始化计数器，用于记录是否包含特定金属字符

        # 遍历特定字符列表
        for mof_metal in mof_metal_elements:
            # 如果字符串包含特定字符，则增加计数器
            if mof_metal in split_part:           #如果当前部分包含金属元素
               contains_target_char = True        #设置计数器为True
               metal_mof = mof_metal              #记录当前金属元素
               break                              #退出内层循环

        # 如果计数器不等于3，表示字符串没有包含所有三个特定字符
        # 将该字符串添加到新列表中
        if not contains_target_char:       #判断如果不包含特定金属元素
           new_list.append(split_part)     #将该部分添加到new_list中
    linker = '.'.join(new_list)            #将new_list中的部分通过'.'连接成一个新的linker字符串
    # print(linker)
    # print(metal_mof)
    linker_dict['linker'].append(linker)   #将新生成的linker添加到linker_dict中的'linker'列表
    linker_dict['metal'].append(metal_mof) #将识别到的金属元素添加到linker_dict中的'metal'列表
    i += 1     #计数器自增1
    print(i)   #打印当前处理的行数
      # linker_dict['linker'].append(new_list)
      # print(new_list)
      # print(len(new_list))
      #
#     # for mof_ori in mof_ori_elements:
#     #     count = smiles_ele.count(mof_ori)
#     #     count_dict[mof_ori].append(count)
#     #     # print(mof_ori,count)
#     #     smiles_ele = smiles_ele.replace(mof_ori, ' ')
#     #
#     # # 官能团统计
#     # mof_gnt = ['c1ccncc1', '[nh2]', '[nh]', '[oh]']
#     # # print(smiles_gnt)
#     # for mof_gnt_ele in mof_gnt:
#     #     count = smiles_gnt.lower().count(mof_gnt_ele)
#     #     smiles_gnt = smiles_gnt.replace(mof_gnt_ele, ' ')
#     #     count_dict[mof_gnt_ele].append(count)
#     #     # print(mof_gnt_ele,count)
#     # metal_group = ['[Zn][O]([Zn])([Zn])[Zn]', '[Cu][Cu]', '[V]', '[Zn][Zn]', '[O]12[Zr]34[O]5[Zr]62[O]2[Zr]71[O]4[Zr]14[O]3[Zr]35[O]6[Zr]2([O]71)[O]43']
#     # metal_simple = 'other'
#     # for metal_sort in metal_group:
#     #     if metal_sort in smiles_metal:
#     #         metal_simple = metal_sort
#     #
#     # count_dict['metal'].append(metal_simple)
#
#
data = pd.DataFrame(linker_dict)
data.to_csv('sample_linker_2.5.csv', index=False)