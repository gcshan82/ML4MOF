import os
from openbabel import openbabel

# 输入和输出目录
input_dir = './MOF_ML/Core-MOF-2014/'
output_dir = './MOF_ML/Core-MOF-2014-sdf/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 初始化 Open Babel 对象
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("cif", "sdf")

# 遍历输入目录中的所有 .cif 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.cif'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.cif', '.sdf'))

        # 读取和转换文件
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, input_path)
        obConversion.WriteFile(mol, output_path)

        print(f"Converted {input_path} to {output_path}")