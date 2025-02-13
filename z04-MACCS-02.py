from rdkit import Chem
from rdkit.Chem import MACCSkeys
import os
import pandas as pd

success_count = 0  # 初始化成功计数器
fail_count = 0     # 初始化失败计数器
data = []

# 指定 SDF 文件路径
directory = './MOF_ML/00-structure3-sdf/'

# 遍历目录中的所有文件
for filename in os.listdir(directory):                    # os.listdir()列出指定目录中的所有文件和文件夹
    if filename.endswith('.sdf'):                         # 筛选以.sdf结尾的文件
        sdf_file = os.path.join(directory, filename)      # 构建每个sdf文件的完整路径
        print(f'Processing file: {filename}')

        # 读取sdf文件
        supplier = Chem.SDMolSupplier(sdf_file)           # 使用RDKIT读取sdf文件(注意：supplier每次仅存储一个sdf文件)

        for i, mol in enumerate(supplier):                # 此处i始终为0
            if mol is not None:
                # 计算 MACCS 分子指纹
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                data.append(list(maccs_fp))  # 非空时正常添加

                # 打印分子编号和指纹信息
                print(f' Molecule {i} MACCS Fingerprint: {list(maccs_fp)}')
                success_count += 1  # 成功计数器加一
                print(f'Total successful fingerprints: {success_count};  Total failed fingerprints: {fail_count}.')

            else:
                data.append([None] * 167)  # 假设你希望表格有5列，所以填充5个 None
                print(f' Molecule {i} could not be read.')
                fail_count += 1    # 失败计数器加一
                print(f'Total successful fingerprints: {success_count};  Total failed fingerprints: {fail_count}.')

df = pd.DataFrame(data)
df.to_excel('./MOF_ML/maccs_fingerprints_multi_row.xlsx', index=False, header=False)
print("数据已成功写入 maccs_fingerprints_multi_row.xlsx 文件")