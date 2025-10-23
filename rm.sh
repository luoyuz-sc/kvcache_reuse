# 进入目标目录
cd ./precompute/kvcache_train

# 删除 5000 到 9999 的 .pt 文件
for i in {1..3000}; do
    file=$(printf "%06d.pt" $i)
    if [ -f "$file" ]; then
        rm "$file"
        echo "Deleted: $file"
    fi
done