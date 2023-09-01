# 读取文件名列表
while IFS= read -r filename; do
    cmd1="python mtaylor_main.py --dataset datasets/feynman/train/$filename"
    $cmd1
    echo "执行文件: $filename"
done < feynman_txt_p1.csv

echo "所有文件执行完成"