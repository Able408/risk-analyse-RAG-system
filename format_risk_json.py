#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re

def format_json_file(input_file, output_file=None):
    """
    读取JSON文件，格式化其中的换行符，并写回文件
    
    参数:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径，默认覆盖输入文件
    """
    if output_file is None:
        output_file = input_file
    
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析JSON文件 {input_file}. 错误信息: {e}")
        return
    except Exception as e:
        print(f"错误: 无法读取文件 {input_file}. 错误信息: {e}")
        return
    
    # 递归处理字典中的所有字符串值，将"\n"替换为真实换行
    def process_value(value):
        if isinstance(value, str):
            # 替换字符串中的"\n"为实际的换行符并添加适当的缩进
            if '\\n' in value:
                lines = value.split('\\n')
                # 对多行内容进行缩进处理
                indented_lines = [lines[0]]  # 第一行不缩进
                for line in lines[1:]:
                    if line.strip():  # 只处理非空行
                        indented_lines.append(line)
                return '\n'.join(indented_lines)
            return value
        elif isinstance(value, dict):
            # 递归处理字典
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # 递归处理列表
            return [process_value(item) for item in value]
        else:
            # 其他类型直接返回
            return value
    
    # 处理整个数据结构
    formatted_data = process_value(data)
    
    # 写回格式化后的JSON，使用较大的缩进值以提高可读性
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=4)
        print(f"成功: 已将格式化后的JSON写入 {output_file}")
    except Exception as e:
        print(f"错误: 无法写入文件 {output_file}. 错误信息: {e}")
        
    # 额外创建一个纯文本版本，便于查看
    try:
        text_output = output_file.rsplit('.', 1)[0] + '_text.txt'
        with open(text_output, 'w', encoding='utf-8') as f:
            for item in formatted_data:
                f.write(f"图片: {item['image']}\n")
                f.write("="*50 + "\n")
                
                result = item['result']
                if 'description' in result:
                    f.write("描述:\n")
                    f.write(result['description'] + "\n\n")
                
                if 'violations' in result:
                    f.write("风险分析:\n")
                    f.write(result['violations'] + "\n")
                
                if 'detail' in result:
                    f.write("错误详情:\n")
                    f.write(result['detail'] + "\n")
                
                if 'error' in result:
                    f.write("错误:\n")
                    f.write(result['error'] + "\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"成功: 已生成纯文本版本 {text_output} 便于阅读")
    except Exception as e:
        print(f"错误: 无法生成纯文本版本. 错误信息: {e}")

if __name__ == "__main__":
    import sys
    
    # 默认处理当前目录下的risk.json文件
    input_file = "risk.json"
    output_file = "risk_formatted.json"
    
    # 如果提供了命令行参数，则使用参数指定的文件
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"正在处理文件: {input_file}")
    format_json_file(input_file, output_file)
    print(f"处理完成. 结果已保存到: {output_file}") 