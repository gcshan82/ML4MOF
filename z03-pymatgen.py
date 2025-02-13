from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# 读取 CIF 文件
cif_file = "./MOF_ML/Core-MOF-2014-test/ABAVIJ_clean.cif"  # 替换为你的 CIF 文件路径
structure = Structure.from_file(cif_file)

# 提取分子信息（如果有）
# 这里假设 CIF 文件描述的是单个分子，否则需要根据实际情况提取分子或原子坐标
# 生成 Pymatgen 分子对象
molecule = structure.get_molecule()  # 如果需要处理单个分子，用该函数

# 将 Pymatgen 的分子对象转换为 RDKit 分子对象
rdkit_mol = Chem.MolFromSmiles(molecule.to('xyz'))  # 使用 SMILES 或 XYZ 格式来转换

# 生成 MACCS 分子指纹
if rdkit_mol:
    maccs_fp = MACCSkeys.GenMACCSKeys(rdkit_mol)
    print(f'MACCS Fingerprint: {list(maccs_fp)}')
else:
    print('Error: Could not convert the molecule to an RDKit object.')