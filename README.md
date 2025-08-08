# risk-analyse-RAG-system
# 项目描述
该项目是一个由多模态大模型驱动的自动化解决方案，旨在从现场图像中实时识别工业安全风险，并自动检索、关联相关的风险描述，最终生成专业的风险分析报告和管控措施。
# 项目概览（RAG部分）
![图片](./rag.svg)
# 部署指南

## 1.环境配置

```
conda create -n flash python=3.10
```

```
conda activate flash
#下载transformer库，解压，安装，在transformer目录下执行 #https://github.com/huggingface/transformers/tree/7a25f8dfdba4c710d278d8312ef2522c5996a894
pip install -e .
```

```
pip install gradio==5.4.0 gradio_client==1.4.2 qwen-vl-utils==0.0.10 transformers-stream-generator==0.0.4  accelerate av
```

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

```
pip install flash-attn==2.6.1 --no-build-isolation
```

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain langchain_community langchain_core sentence-transformers faiss-cpu pypdf fastapi uvicorn python-multipart Pillow
pip install --upgrade autoawq
pip install chardet openpyxl
```

## 2.下载模型

```
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ
modelscope download --model AI-ModelScope/bge-base-zh-v1.5
```

## 3.创建知识库索引

将《风险源清单.xlsx》文件上传到服务器目录kb_risk

```
#创建向量数据库
python vector1.py
```

这会在指定的 VECTOR_STORE_PATH 生成 FAISS 索引文件。

## 4.运行RAG脚本

将“风险源列表_清洗后.txt”上传至服务器

```
python qwen_32b_flash.py
```

![image-20250728105339778](C:\Users\wrs\AppData\Roaming\Typora\typora-user-images\image-20250728105339778.png)

## 5.测试

将测试图片上传至服务器

```
bash analyze_images.sh
```

格式优化

```
python format_risk_json.py
```

输出为doc文档

```
python generate_doc_with_images.py
```

