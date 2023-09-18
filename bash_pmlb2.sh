# 读取文件名列表
while IFS= read -r filename; do
    cmd1="python mtaylor_main.py --trainset datasets/pmlb/train/$filename.txt --varset datasets/pmlb/val_csv/$filename.csv"
    $cmd1
    echo "执行文件: $filename"
done < pmlb2.csv

echo "所有文件执行完成"