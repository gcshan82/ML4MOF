import os

# 设置目录路径
relabel_directory = "./01-MOFid/00-structure3"
core_mofid_file = "./01-MOFid/core_mofid.txt"
output_file = "./01-MOFid/output.txt"  # 输出文件名

# 读取 core_mofid 文件内容
with open(core_mofid_file, 'r') as f:
    core_mofid_lines = f.readlines()

# 创建输出文件
with open(output_file, 'w') as out_file:
    not_found_count = 0  # 计数器
    # 遍历 re-label 目录下的所有文件
    for filename in os.listdir(relabel_directory):
        if filename.endswith(".cif"):
            # 删除 .cif 后缀
            name_without_extension = filename[:-4]
            # 检查 core_mofid 文件中是否包含该文件名
            found = False
            for line in core_mofid_lines:
                if name_without_extension in line:
                    out_file.write(line)  # 将匹配的行写入输出文件
                    found = True
                    break
            # 如果未找到，则打印文件名
            if not found:
                print(f"未找到: {filename}")
                not_found_count += 1

    # 输出未找到的文件总数
    print(f"未找到的文件总数: {not_found_count}")