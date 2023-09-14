# 读取文件名列表
while IFS= read -r filename; do
    cmd1="python write_in_feature.py --dataset $filename"
    $cmd1
    echo "执行文件: $filename"
done < name.csv

echo "所有文件执行完成"