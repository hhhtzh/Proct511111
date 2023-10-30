# 读取文件名列表
while IFS= read -r filename; do
    cmd1="python mtaylor.py --name $filename"
    # cmd1="python mtaylor_main.py --trainset datasets/feynman/train/$filename.txt --varset datasets/feynman/mydataver/$filename.csv"
    $cmd1
    echo "执行文件: $filename"
done < feynman_txt_p1.csv

echo "所有文件执行完成"