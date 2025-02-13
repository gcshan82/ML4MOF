import os
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pandas as pd

# 指定输入和输出目录
input_dir = './MOF_ML/Core-MOF-2014-test/'
output_dir = './MOF_ML/01-MACCS/'

# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 列出所有的 CIF 文件
cif_files = [f for f in os.listdir(input_dir) if f.endswith('.cif')]

# 创建一个空列表来保存分子指纹
fingerprints_list = []

# 创建 OpenBabel 转换器对象
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("cif", "sdf")

for cif_file in cif_files:
    # 输入和输出文件路径
    input_path = os.path.join(input_dir, cif_file)
    sdf_path = os.path.join(output_dir, cif_file.replace('.cif', '.sdf'))

    # 创建一个 OBMol 对象用于存储分子
    obMol = openbabel.OBMol()

    # 尝试读取和转换 CIF 文件
    try:
        if obConversion.ReadFile(obMol, input_path):
            # 检查是否成功读取分子
            if obMol.NumAtoms() == 0:
                print(f"No atoms found in {input_path}, skipping conversion.")
                continue

            # 写入 SDF 文件
            if obConversion.WriteFile(obMol, sdf_path):
                print(f"Successfully converted {input_path} to {sdf_path}")
            else:
                print(f"Error writing SDF file: {sdf_path}")
        else:
            print(f"Error reading CIF file: {input_path}")

    except Exception as e:
        print(f"Exception occurred during conversion of {input_path} to {sdf_path}: {str(e)}")
        continue

    # 检查 SDF 文件是否生成，并尝试读取
    if os.path.exists(sdf_path):
        suppl = Chem.SDMolSupplier(sdf_path)
        for mol in suppl:
            if mol is not None:
                # 计算 MACCS 分子指纹
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)

                # 将指纹转换为列表并添加到列表中
                fingerprints_list.append(list(maccs_fp.ToBitString()))
            else:
                print(f"Failed to read molecule from {sdf_path}")

# 将列表转换为 DataFrame
fingerprints_df = pd.DataFrame(fingerprints_list)

# 保存分子指纹为 CSV 文件
fingerprints_df.to_csv(os.path.join(output_dir, 'maccs_fingerprints.csv'), index=False)

print('分子指纹已保存！')