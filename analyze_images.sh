#!/bin/bash

# 检查jq是否已安装
if ! command -v jq &> /dev/null; then
    echo "错误: 需要安装jq工具来处理JSON"
    echo "请运行: sudo apt-get install jq"
    exit 1
fi

# 创建输出目录和文件
OUTPUT_FILE="risk.json"
TEMP_DIR="tmp_json"

# 创建临时目录
mkdir -p $TEMP_DIR

# 获取test_image目录下的所有PNG图片
IMAGE_DIR="test_image"
IMAGES=$(find $IMAGE_DIR -name "*.png")

# 创建结果数组
echo "[]" > "$TEMP_DIR/results.json"

# 遍历每一张图片
for image in $IMAGES; do
  echo "处理图片: $image"
  
  # 使用curl发送请求
  curl -s -X POST "http://localhost:8080/analyze_image_local" \
       -F "file=@$image" \
       -H "accept: application/json" > "$TEMP_DIR/response.json"
  
  # 获取图片名称
  image_name=$(basename "$image")
  
  # 检查API响应格式
  if jq empty "$TEMP_DIR/response.json" 2>/dev/null; then
    # 处理响应中的换行符
    jq --arg img "$image_name" '
      # 创建基本对象结构
      {image: $img, result: .} |
      
      # 处理结果中的violations字段
      if .result.violations then
        .result.violations = (.result.violations | gsub("\\\\n"; "\n"))
      else . end |
      
      # 处理结果中的description字段
      if .result.description then
        .result.description = (.result.description | gsub("\\\\n"; "\n"))
      else . end
    ' "$TEMP_DIR/response.json" > "$TEMP_DIR/formatted_item.json"
  else
    # API返回了非法的JSON，创建一个错误对象
    echo "{\"image\": \"$image_name\", \"result\": {\"error\": \"无效的JSON响应\"}}" > "$TEMP_DIR/formatted_item.json"
  fi
  
  # 将当前结果添加到总结果
  jq -s '.[0] + [.[1]]' "$TEMP_DIR/results.json" "$TEMP_DIR/formatted_item.json" > "$TEMP_DIR/temp.json"
  mv "$TEMP_DIR/temp.json" "$TEMP_DIR/results.json"
  
  echo "完成处理: $image_name"
done

# 美化并输出最终结果
jq '.' "$TEMP_DIR/results.json" > $OUTPUT_FILE

# 清理临时文件
rm -rf $TEMP_DIR

echo "分析完成！结果已保存到 $OUTPUT_FILE，格式已优化，换行已处理" 