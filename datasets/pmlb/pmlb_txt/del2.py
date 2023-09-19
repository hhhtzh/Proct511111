import os

def delete_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_lines = len(lines)
    num_lines_to_delete = num_lines // 4

    new_lines = lines[num_lines_to_delete:]

    with open(file_path, 'w') as file:
        file.writelines(new_lines)

def delete_lines_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            delete_lines(file_path)

# 指定要处理的文件夹路径
folder_path = '/home/tzh/下载/pmlb_txt'

# 删除文件夹中所有 txt 文件的前四分之一行
delete_lines_in_folder(folder_path)
