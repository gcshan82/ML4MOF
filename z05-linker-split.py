import re

#该脚本针对00-mofids.txt文件
def extract_smiles_linkers(file_path):   #输出每个MOFid的所有linker smiles
    # 打开输入文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 打开输出文件
        with open('./02-feature-expland/extracted_smiles_linkers.txt', 'w', encoding='utf-8') as output_file:
            line_number = 1  # 初始化行号
            # 逐行读取文件
            for line in file:
                # 查找 'smiles_linkers': 后， 'smiles' 之前的部分
                match = re.search(r"'smiles_linkers':\s*\[(.*?)\]\s*,\s*'smiles'", line)
                if match:
                    # 提取''中的内容
                    linkers_str = match.group(1)
                    # 使用正则表达式提取单个linker
                    linkers = re.findall(r"'([^']+)'", linkers_str)
                    # 将提取到的linkers使用空格拼接
                    combined_linkers = ' '.join(linkers)
                    # 将拼接后的字符串写入输出文件
                    output_file.write(f"{line_number} {combined_linkers}\n")
                    print(f"{line_number}: {combined_linkers}")  # 输出拼接后的字符串
                    line_number += 1  # 增加行号


def extract_smiles_long_linker(file_path):    #输出每个MOFid中最长的那个linker smiles
    # 打开输入文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 打开输出文件
        with open('./02-feature-expland/extracted_smiles_long_linkers.txt', 'w', encoding='utf-8') as output_file:
            line_number = 1  # 初始化行号
            # 逐行读取文件
            for line in file:
                # 查找 'smiles_linkers': 后， 'smiles' 之前的部分
                match = re.search(r"'smiles_linkers':\s*\[(.*?)\]\s*,\s*'smiles'", line)
                if match:
                    # 提取 'smiles_linkers' 部分的内容
                    linkers_str = match.group(1)
                    # 使用正则表达式提取单个linker
                    linkers = re.findall(r"'([^']+)'", linkers_str)
                    if linkers:  # 如果存在linkers
                       # 找到最长的linker
                       longest_linker = max(linkers, key=len)
                       # 将行号和最长的linker写入输出文件
                       output_file.write(f"{line_number} {longest_linker}\n")
                       print(f"{line_number}: {longest_linker}")  # 输出最长的linker
                       line_number += 1  # 增加行号
                    else:
                        # 将行号和最长的linker写入输出文件
                        output_file.write(f"{line_number}\n")
                        print(f"{line_number}")
                        line_number += 1  # 增加行号



# 处理00-mofids.txt文件，得到linker的smiles
#extract_smiles_linkers('./02-feature-expland/00-mofids.txt')
extract_smiles_long_linker('./02-feature-expland/00-mofids.txt')