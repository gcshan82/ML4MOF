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



# pressure_list = ['001', '005', '01', '05', '25']
# pressure_list = ['001']
# # discriptor_list = ['calculated', 'chem', 'struc']
#
#
# discriptor_list = ['calculated', 'chem', 'struc']
# discriptor_list = ['calculated']
#
# for discriptor_single in discriptor_list:
#     for pressure_single in tqdm(pressure_list):
#
#         data = pd.read_csv('./ml_record/abs/'+discriptor_single+'/y_'+pressure_single+'_clean.csv')
#         sns.jointplot(x=data['y_test'], y=data['y_predict'], #设置xy轴，显示columns名称
#                       data = data,  #设置数据
#                       color = '#87CEFA', #设置颜色
#                       #s = 50, edgecolor = 'w', linewidth = 1,#设置散点大小、边缘颜色及宽度(只针对scatter)
#                       # stat_func=sci.pearsonr,
#                       kind = 'hex',#设置类型：'scatter','reg','resid','kde','hex'
#                       space = 0.1, #设置散点图和布局图的间距
#                       height = 8, #图表大小(自动调整为正方形))
#                       ratio = 5, #散点图与布局图高度比，整型
#                       # marginal_kws = dict(bins=15, rug =True) #设置柱状图箱数，是否设置rug
#                       )
#         # plt.show()
#         plt.savefig('./ml_record/abs/'+discriptor_single+'/y_'+pressure_single+'.png', dpi=300,transparent=True)


discriptor_list = ['calculated', 'chem', 'struc']

data = pd.read_csv('./ml_record/abs/' + 'struc' + '/y_' + '001' + '_clean.csv')
jointgrid = sns.jointplot(x=data['y_test'], y=data['y_predict'],  # 设置xy轴，显示columns名称
            data=data,  # 设置数据
            color='#87CEFA',  # 设置颜色
            # s = 50, edgecolor = 'w', linewidth = 1,#设置散点大小、边缘颜色及宽度(只针对scatter)
            # stat_func=sci.pearsonr,
            kind='hex',  # 设置类型：'scatter','reg','resid','kde','hex'
            space=0.1,  # 设置散点图和布局图的间距
            height=8,  # 图表大小(自动调整为正方形))
            ratio=5,  # 散点图与布局图高度比，整型
            # marginal_kws = dict(bins=15, rug =True) #设置柱状图箱数，是否设置rug
                      )
# ax.set_axis_labels(x_label, y_label, fontsize=20)
# 获取 x 轴刻度位置和标签



plt.savefig('./ml_record/abs/'+'struc'+'/y_'+'001'+'.png', dpi=300,transparent=True)
plt.show()
# marginal_kws = dict(bins=15, rug =True) #设置柱状图箱数，是否设置rug





# 12
#
#     mol = Chem.MolFromSmiles('[O-]C(=O)C#Cc1cc(Br)c(c(c1Br)Br)C#CC(=O)[O-].[O-]C(=O)C#Cc1ccc(c(c1)Br)C#CC(=O)[O-].[O-]C(=O)C#Cc1ccc(cc1)C#CC(=O)[O-]')
# # descriptor_names = [x[0] for x in Descriptors._descList]
# # descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
# # descriptors = pd.DataFrame(
# #     [descriptor_calculator.CalcDescriptors(mol)],
# #     columns=descriptor_names
# # )
# # print(descriptors)
#
#     calc = Calculator(descriptors, ignore_3D=True)
#     X_mord = pd.DataFrame(calc.pandas([mol]))
#     print(X_mord)


#
# input_file = 'hMOF_CO2_2.5_small_mofid.csv'
# output_file = 'output_large.csv'
# data = pd.read_csv(input_file)
# count_dict = {'cl': [], 'o': [], 'n': [], 'br': [], 'c': [], 'f': [], 'c1ccncc1': [], '[nh2]': [], '[nh]': [], '[oh]': [], 'structure': [], 'cat': [], 'metal': [], 'label': []}
# linker_dict = {'linker': [], 'metal': [], 'label': []}
# str_list = []
# i = 0
# # 使用split函数按照'.'进行分割
# # 有机配体元素统计
# for index, row in data.iterrows():
#     smiles = row['smiles']
#     label = row['label']
#     count_dict['label'].append(label)
#     separator = '&&'
#     smiles_ele = smiles.split(separator, 1)[0]#金属位点加有机配体
#     smiles_str = smiles.split(separator, 1)[1]#拓扑结构与cat
#     smiles_gnt = smiles_ele
#     smiles_metal = smiles_ele
#     linker_list = []
#     new_list = []
#
#     linker_dict['label'].append(label)
#
#     split_parts = smiles_ele.split('.')#按‘.'进行切分
#     mof_metal_elements = ['Zn', 'Cu', 'V', 'Zr']
#     mof_ori_elements = ['cl', 'o', 'n', 'br', 'c', 'f']
#
#
#     for split_part in split_parts:
#         # 初始化一个计数器，用于记录包含特定字符的数量
#         contains_target_char = False
#
#         # 遍历特定字符列表
#         for mof_metal in mof_metal_elements:
#             # 如果字符串包含特定字符，则增加计数器
#             if mof_metal in split_part :
#                 contains_target_char = True
#                 metal_mof = mof_metal
#                 break
#
#         # 如果计数器不等于3，表示字符串没有包含所有三个特定字符
#         # 将该字符串添加到新列表中
#         if not contains_target_char:
#             new_list.append(split_part)
#     linker = '.'.join(new_list)
#     # print(linker)
#     # print(metal_mof)
#     linker_dict['linker'].append(linker)
#     linker_dict['metal'].append(metal_mof)
#     i += 1
#     print(i)
#     # linker_dict['linker'].append(new_list)
#     # print(new_list)
#     # print(len(new_list))
#     #
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
# data = pd.DataFrame(linker_dict)
# data.to_csv('sample_linker_2.5.csv', index=False)