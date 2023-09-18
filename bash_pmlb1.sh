# 读取文件名列表
while IFS= read -r filename; do
    cmd1="python mtaylor_main.py --trainset datasets/pmlb/train/$filename.txt --varset datasets/pmlb/val/$filename.csv"
    $cmd1
    echo "执行文件: $filename"
done < pmlb1.csv

echo "所有文件执行完成"