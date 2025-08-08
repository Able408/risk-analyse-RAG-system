# create_vector_db_table.py
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import chardet

# --- 配置 ---
KNOWLEDGE_BASE_DIR = "/mnt/data/wrs/kb_risk"  # 存放表格文档的目录 (csv, xlsx, txt等)
EMBEDDING_MODEL_NAME = "/mnt/data/wrs/bge-base-zh-v1.5"
VECTOR_STORE_PATH = "vectorstore/risk_faiss_index_confined_space"
# --- 配置结束 ---

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def load_table_documents(directory):
    """加载表格文档并按行处理"""
    documents = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            try:
                df = None
                
                if file_ext == '.csv':
                    # 检测CSV文件编码
                    encoding = detect_encoding(file_path)
                    df = pd.read_csv(file_path, encoding=encoding)
                
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                
                elif file_ext == '.txt':
                    # 假设是制表符或逗号分隔的文本文件
                    encoding = detect_encoding(file_path)
                    # 先尝试制表符分隔
                    try:
                        df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                        if len(df.columns) == 1:  # 如果只有一列，尝试逗号分隔
                            df = pd.read_csv(file_path, sep=',', encoding=encoding)
                    except:
                        df = pd.read_csv(file_path, sep=',', encoding=encoding)
                
                if df is not None and not df.empty:
                    # 处理每一行
                    documents.extend(process_table_rows(df, file))
                    print(f"成功处理文件: {file}, 共 {len(df)} 行")
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                continue
    
    print(f"总共加载了 {len(documents)} 个文档片段。")
    return documents

def process_table_rows(df, source_file):
    """将DataFrame的每一行转换为Document对象"""
    documents = []
    
    # 清理列名
    df.columns = df.columns.astype(str).str.strip()
    
    for index, row in df.iterrows():
        # 方式1: 结构化文本格式
        content_parts = []
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():  # 过滤空值
                content_parts.append(f"{col}: {str(value).strip()}")
        
        if content_parts:  # 只有当行中有有效内容时才创建文档
            content = " | ".join(content_parts)
            
            # 创建元数据
            metadata = {
                "source": source_file,
                "row_index": index,
                "type": "table_row"
            }
            
            # 添加重要字段到元数据（方便后续查询和过滤）
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    # 将列名转换为有效的元数据键
                    clean_col = col.replace(" ", "_").replace("（", "").replace("）", "").replace("(", "").replace(")", "")
                    metadata[f"field_{clean_col}"] = str(value).strip()
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
    
    return documents

def create_enhanced_content(row):
    """为表格行创建增强的文本内容，便于语义搜索"""
    content_parts = []
    
    # 根据你的表格结构，可以自定义重要字段的处理
    important_fields = ["风险源名称", "场所位置", "行业领域", "序号"]  # 根据实际情况调整
    
    # 优先处理重要字段
    for field in important_fields:
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            content_parts.append(f"{field}: {str(row[field]).strip()}")
    
    # 处理其他字段
    for col, value in row.items():
        if col not in important_fields and pd.notna(value) and str(value).strip():
            content_parts.append(f"{col}: {str(value).strip()}")
    
    return " | ".join(content_parts)

def create_and_save_vector_db(documents):
    """创建并保存向量数据库"""
    if not documents:
        print("没有文档可以创建向量数据库")
        return
    
    print(f"使用嵌入模型: {EMBEDDING_MODEL_NAME}")
    print(f"准备创建向量数据库，共 {len(documents)} 个文档片段")
    
    # 显示几个样例
    print("\n样例文档内容:")
    for i, doc in enumerate(documents[:3]):
        print(f"文档 {i+1}: {doc.page_content[:200]}...")
        print(f"元数据: {doc.metadata}")
        print("-" * 50)
    
    model_kwargs = {}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("开始创建向量数据库 (FAISS)...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("向量数据库创建完成。")

    # 创建保存目录
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"向量数据库已保存到: {VECTOR_STORE_PATH}")

def test_vector_db():
    """测试向量数据库"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # 测试查询
        test_query = "危险化学品"
        results = vectorstore.similarity_search(test_query, k=3)
        
        print(f"\n测试查询: '{test_query}'")
        print("查询结果:")
        for i, doc in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"内容: {doc.page_content}")
            print(f"来源: {doc.metadata.get('source', 'Unknown')}")
            print(f"行号: {doc.metadata.get('row_index', 'Unknown')}")
            
    except Exception as e:
        print(f"测试向量数据库时出错: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"错误：知识库目录 '{KNOWLEDGE_BASE_DIR}' 不存在。请创建该目录并将表格文件放入其中。")
    else:
        # 创建向量数据库
        docs = load_table_documents(KNOWLEDGE_BASE_DIR)
        if docs:
            create_and_save_vector_db(docs)
            # 测试向量数据库
            test_vector_db()
        else:
            print("未加载到任何文档，请检查目录和文件。")