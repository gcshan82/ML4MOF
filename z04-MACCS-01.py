from rdkit import Chem
from rdkit.Chem import MACCSkeys

# 读取 SDF 文件
sdf_file_path = './MOF_ML/Core-MOF-2014-test-sdf/ABAVIJ_clean.sdf'

# 使用 RDKit 读取 SDF 文件
supplier = Chem.SDMolSupplier(sdf_file_path)

# 遍历文件中的每个分子
for mol in supplier:
    if mol is not None:
        # 计算 MACCS 分子指纹
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)

        # 打印 MACCS 分子指纹
        print(list(maccs_fp))
    else:
        print("遇到无效的分子，跳过。")