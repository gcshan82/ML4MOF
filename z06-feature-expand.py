import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import rdBase, Chem   # rdkit: 开源的化学信息学工具包; reBase: 基础模块; Chem: 可进行分子结构的读取、表示、描述符计算等
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from collections import Counter # Python库collections模块中Counter类: 进行技术操作和频率统计
from mordred import Calculator, descriptors


#对metal进行扩充
def metal_expand(file_path):      #将输入文件中的金属元素，按照elemental_descriptors.csv扩展为6列，最终输出
    input_file = file_path
    output_file = 'output_large.csv'
    data = pd.read_csv(input_file)
    elements = pd.read_csv('./02-feature-expland/elemental_descriptors.csv')      # 读取元素相关性质(包括原子序数、重量、半径、密立根电荷、极化率和电子亲和性等)
    # 创建新的DataFrame并定义6个列名，metal_features_all在循环中不断更新增加新的内容
    metal_features_all = pd.DataFrame(columns=['Atomic_Number', 'Atomic_Weight', 'Atomic Radius', 'Mulliken EN',
                                                   'polarizability(A^3)', 'electron affinity(kJ/mol)'])
    # 初始化计数器
    loop_counter = 0

    for index, row in data.iterrows():      #使用iterrows方法逐行遍历data DataFrame，每行数据以row变量存储(循环次数与行数相同)
        #linker = row['linker']              #当前行中提取linker列的值
        #label = row['label']
        metal = row['metal2']
        # print(metal)
        metal_test = elements.loc[elements['Symbol'] == metal].copy()    #elements中查找符号等于当前metal变量的行，并创建副本赋值给metal_list
        # print(metal_test)

        # 创建新的DataFrame，数据来源于metal_test，列为金属特征的名称
        metal_features = pd.DataFrame(data=metal_test,
                                          columns=['Atomic_Number', 'Atomic_Weight', 'Atomic Radius', 'Mulliken EN',
                                                   'polarizability(A^3)', 'electron affinity(kJ/mol)'])
        #metal_features['Row_Index'] = row.name  # 将当前行号加入为新列
        # 增加计数器
        loop_counter += 1
        metal_features['ID'] = loop_counter  # 将循环次数加入为新列
        metal_features.reset_index(drop=True, inplace=True)               #重置metal_feature的行索引
        metal_features_all = pd.concat([metal_features_all, metal_features], ignore_index=True)
    print(metal_features_all)
    metal_features_all.to_csv('./02-feature-expland/metal-expand2.csv', index=False)


#对linker进行扩充，输入文件为test_linker.csv，输出文件为test_mordred.csv
def linker_expand(file_path):
    input_file = file_path                 #input_file = 'test_linker.csv'
    data = pd.read_csv(input_file)
    i = 1
    for index, row in data.iterrows():     # 遍历data DataFrame的每一行，获取索引和行内容
        linker = row['linker']             # 获取当前行中'linker'列的值
        # 检查linker是否为空或无效的
        if pd.isnull(linker) or linker.strip() == '' or isinstance(linker, float):
            print(f"Invalid SMILES at index {i}: {linker}")
            # 创建一行空行，列名与 mordred_features_all 相同
            empty_row = pd.DataFrame([[None] * len(mordred_features_all.columns)], columns=mordred_features_all.columns)
            # 将空行添加到 mordred_features_all
            mordred_features_all = pd.concat([mordred_features_all, empty_row], ignore_index=True)
            i += 1
            continue
        mol=Chem.MolFromSmiles(linker)     # 使用RDKit将linker的SMILES表示转换为分子对象mol
        # 检查分子对象是否成功创建
        if mol is None:
            print(f"Failed to create molecule from SMILES at index {i}: {linker}")
            # 创建一行空行，列名与 mordred_features_all 相同
            empty_row = pd.DataFrame([[None] * len(mordred_features_all.columns)], columns=mordred_features_all.columns)
            # 将空行添加到 mordred_features_all
            mordred_features_all = pd.concat([mordred_features_all, empty_row], ignore_index=True)
            i += 1
            continue
        calc = Calculator(descriptors, ignore_3D=True)    # 创建Calculator对象，用于计算分子描述符，忽略三维信息
        X_mord = pd.DataFrame(calc.pandas([mol]), dtype=float)    # 计算分子的描述符，并将结果转换为DataFrame，存储在X_mord中
        # 插入 `i` 作为X_mord的第一列
        X_mord.insert(0, 'index', i)
        print(X_mord)
        print(f"ID {i} is over")    # 使用 f-string 来直接插入变量
        if i == 1:
            mordred_features_all = X_mord     # 将X_mord赋值给mordred_features_all
        if i > 1:
            mordred_features_all = pd.concat([mordred_features_all, X_mord])   # 将新的描述符DataFrame与已有的合并
        i += 1
    mordred_features_all.to_csv('./02-feature-expland/linker_mordred.csv', index=False)
    print(mordred_features_all)


def choose_RDkit(file_path):              #提取无法被RDkit读取的MOF id
    input_file = file_path
    data = pd.read_csv(input_file)
    descriptors_mord = {}
    error_list = []
    i = 2
    for index, row in data.iterrows():
        linker = row['linker']
        # print(metal)
        mol = Chem.MolFromSmiles(linker)

        if mol == None:
            print(i)
            error_list.append(i)
        i += 1


def lenth_choose(file_path):             #将只有1个linker的MOFid行索引提取出来
    input_file = file_path
    data = pd.read_csv(input_file)
    lenth_1 = []
    i = 0
    for index, row in data.iterrows():
        linker = row['linker']
        split_parts = linker.split('.')  #按‘.'进行切割，结果存储在split_parts列表中
        if len(split_parts) == 1:
            lenth_1.append(index)        #将当前索引添加到lenth_1列表中
            i += 1
            print(i)
    file = open("lenth1.txt", "w")       #以写入模式打开或创建lenth1.txt文件
    for item in lenth_1:                 #遍历lenth_1中的每个元素
        file.write(str(item) + "\n")     #将行索引转换为字符写入文件中
    file.close()


def cal_select(file_path1,file_path2):   #从两个CSV文件中读取数据，找到它们的公共列并存储
    input_file1 = file_path1
    input_file2 = file_path2
    ori_data = pd.read_csv(input_file1)
    name_data = pd.read_csv(input_file2)
    common_columns = set(ori_data.columns) & set(name_data.columns)
    result = ori_data[list(common_columns)]
    result.to_csv('select_result_all.csv', index=False)

#cal_select('test_mordred_all.csv','cal_importance_25.csv')
#metal_expand('./02-feature-expland/metal.csv')
if __name__ == '__main__':
   linker_expand('./02-feature-expland/long-linker.csv')