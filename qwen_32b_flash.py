import base64
import io
import os
import gc
from typing import List, Any, Dict, Optional
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableAssign
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

# --- Configuration ---
# 保持 AWQ 模型路径
MODEL_CHECKPOINT_PATH = '/dev/shm/wrs/Qwen2.5-VL-32B-Instruct-AWQ'

# 禁用 BitsAndBytes 量化（AWQ 模型已经预量化）
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
USE_FLASH_ATTN_2 = True # 启用 Flash Attention
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

VECTOR_STORE_PATH = "vectorstore/risk_faiss_index_confined_space"
EMBEDDING_MODEL_NAME = "/mnt/data/wrs/bge-base-zh-v1.5"
RETRIEVAL_K = 30
MAX_NEW_TOKENS = 2048
REPETITION_PENALTY = 1.2
TEMPERATURE = 0.2
TOP_P = 0.9
RISK_SOURCE_LIST_PATH = "风险源列表_清洗后.txt"
RISK_SOURCE_TOP_K = 2 # 最终需要返回的风险源数量
RISK_SOURCE_CANDIDATE_K = 15 # 为LLM验证准备的候选词数量
RISK_SOURCE_SIMILARITY_THRESHOLD = 0.7

# --- Helper Functions ---
def encode_image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    img_format = pil_image.format if pil_image.format and pil_image.format.upper() in ['JPEG', 'PNG', 'GIF', 'BMP'] else 'JPEG'
    pil_image.save(buffered, format=img_format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert('RGB')

def format_docs(docs: List) -> str:
    return "\n\n".join([f"来源: {doc.metadata.get('source', '未知')}\n内容: {doc.page_content}" for doc in docs])

def dedup_docs_simple(docs: List, max_total_docs: int = 30) -> List:
    seen_content = set()
    unique_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen_content:
            seen_content.add(content)
            unique_docs.append(doc)
            if len(unique_docs) >= max_total_docs:
                break
    return unique_docs

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_and_embed_risk_sources(list_path: str, embedder: HuggingFaceEmbeddings):
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"未找到风险源列表文件: {list_path}")

    with open(list_path, 'r', encoding='utf-8') as f:
        risk_sources = [line.strip() for line in f if line.strip()]

    print(f"已加载 {len(risk_sources)} 条风险源名称，开始计算嵌入……")
    rs_embeddings = embedder.embed_documents(risk_sources)
    rs_embeddings = np.array(rs_embeddings, dtype=np.float32)
    print("风险源嵌入计算完成。")
    return risk_sources, rs_embeddings

# --- Custom LangChain Wrapper for Local Qwen2-VL (No changes here) ---
class CustomQwenVLModel(BaseChatModel):
    model: Qwen2_5_VLForConditionalGeneration
    processor: Qwen2_5_VLProcessor
    device: str
    max_new_tokens: int = MAX_NEW_TOKENS

    class Config:
        arbitrary_types_allowed = True

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        pil_images = []
        messages_for_template = []
        for msg in messages:
            role = "system" if isinstance(msg, SystemMessage) else "assistant" if isinstance(msg, AIMessage) else "user"
            if isinstance(msg.content, str):
                messages_for_template.append({"role": role, "content": msg.content})
            elif isinstance(msg.content, list):
                content_list_for_role = []
                for item in msg.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_list_for_role.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            data_url = image_url.get("url", "")
                            if data_url.startswith("data:image") and ";base64," in data_url:
                                base64_string = data_url.split(";base64,", 1)[1]
                                try:
                                    image = decode_base64_to_image(base64_string)
                                    pil_images.append(image)
                                    content_list_for_role.append({"type": "image"})
                                except Exception as e:
                                    print(f"警告: 无法解码 base64 图片: {e}")
                messages_for_template.append({"role": role, "content": content_list_for_role})
        try:
            text_prompt = self.processor.apply_chat_template(messages_for_template, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"错误: 应用聊天模板失败: {e}\n内容: {messages_for_template}"); raise e
        try:
            inputs = self.processor(text=[text_prompt], images=pil_images if pil_images else None, padding=True, return_tensors='pt').to(self.device)
        except Exception as e:
            print(f"错误: 调用 processor 失败: {e}"); raise e
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P, "repetition_penalty": REPETITION_PENALTY, **kwargs, **inputs}
        gen_kwargs.pop("streamer", None)
        with torch.no_grad():
            try: output_ids = self.model.generate(**gen_kwargs)
            except Exception as e: print(f"错误: 模型生成失败: {e}"); cleanup_memory(); raise e
        input_token_len = inputs['input_ids'].shape[1]
        response_ids = output_ids[:, input_token_len:]
        response_text = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0]
        del inputs, output_ids, response_ids; cleanup_memory()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

    @property
    def _llm_type(self) -> str: return "custom_qwen_vl"

# --- Initialization (No changes until the chains) ---
print(f"正在从以下位置加载模型: {MODEL_CHECKPOINT_PATH}")
print(f"使用设备: {DEVICE}")
print(f"使用 Flash Attention 2: {USE_FLASH_ATTN_2}")
print(f"4位加载: {LOAD_IN_4BIT}, 8位加载: {LOAD_IN_8BIT}")
print(f"重复惩罚系数: {REPETITION_PENALTY}")

quantization_config = None
if LOAD_IN_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
elif LOAD_IN_8BIT:
     quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

# 在模型加载部分，确保不使用 quantization_config
model_args = {
    "device_map": "auto",
    "torch_dtype": torch.float16,  # 强制使用 bfloat16 以兼容 AWQ 量化
    # 移除或注释掉这行："quantization_config": quantization_config,
}

if USE_FLASH_ATTN_2:
    if 'quantization_config' in model_args and model_args['quantization_config'] is not None:
         print("警告: Flash Attention 2 可能与 bitsandbytes 量化不兼容。将不使用 Flash Attn。")
    elif DEVICE == 'cpu':
         print("警告: Flash Attention 2 需要 CUDA。将不使用 Flash Attn。")
    else:
        try:
            import flash_attn
            model_args["attn_implementation"] = "flash_attention_2"
            print("使用 Flash Attention 2")
        except ImportError:
            print("选择了 Flash Attention 2 但未找到。请使用 'pip install flash-attn --no-build-isolation' 安装。将回退。")
            USE_FLASH_ATTN_2 = False

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_CHECKPOINT_PATH,
        **model_args
    )
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT_PATH)
    print(f"实际加载的 Processor 类型: {type(processor)}")

    if quantization_config:
        model.to(DEVICE)
    model.eval()
    print("模型和处理器加载成功。")
except Exception as e:
    print(f"错误: 加载模型/处理器失败: {e}")
    raise SystemExit("无法加载模型。请检查路径、依赖项和内存。")


print(f"正在从以下位置加载向量数据库: {VECTOR_STORE_PATH}")
if not os.path.exists(VECTOR_STORE_PATH):
    raise FileNotFoundError(f"错误：向量数据库索引 '{VECTOR_STORE_PATH}' 未找到。请先运行 create_vector_db.py。")

print(f"正在加载嵌入模型: {EMBEDDING_MODEL_NAME}")
embed_model_kwargs = {'device': DEVICE}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=embed_model_kwargs,
    encode_kwargs=encode_kwargs
)
vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': RETRIEVAL_K, 'fetch_k': 100}
)
print("向量数据库加载成功。")

# 在第220行左右，现有llm初始化后添加
print("正在初始化基础 LLM 包装器...")
try:
    # 基础LLM，使用默认的max_new_tokens
    llm = CustomQwenVLModel(
        model=model, 
        processor=processor, 
        device=DEVICE, 
        max_new_tokens=MAX_NEW_TOKENS  # 使用默认值
    )
    print("基础 LLM 包装器初始化成功。")
except Exception as e:
    print(f"错误: 初始化基础 CustomQwenVLModel 失败: {e}")
    raise

print("正在初始化管理措施专用 LLM 包装器...")
try:
    # 专门用于管理措施生成的LLM，限制输出长度
    management_llm = CustomQwenVLModel(
        model=model, 
        processor=processor, 
        device=DEVICE, 
        max_new_tokens=600  # 限制为600 tokens
    )
    print("管理措施专用 LLM 包装器初始化成功。")
except Exception as e:
    print(f"错误: 初始化管理措施专用 CustomQwenVLModel 失败: {e}")
    raise

risk_source_list, risk_source_embeddings = None, None
try:
    risk_source_list, risk_source_embeddings = load_and_embed_risk_sources(RISK_SOURCE_LIST_PATH, embeddings)
except Exception as e:
    print(f"加载风险源列表失败: {e}，后续将跳过风险源匹配功能。")

# --- Candidate Generation Function (No changes here, it's now a candidate generator) ---
def get_hybrid_risk_sources(text: str, top_k: int, similarity_threshold: float) -> List[str]:
    if risk_source_list is None or risk_source_embeddings is None: return []
    candidates = {}
    stop_words = {"其他", "工具", "床", "刀", "系统", "事故", "装置", "设备", "材料", "部位", "培训"}
    generic_keywords = ("系统", "事故", "培训", "措施", "管理", "制度")
    for item_from_file in risk_source_list:
        item = item_from_file.strip()
        is_generic = any(gk in item for gk in generic_keywords)
        if item and item not in stop_words and len(item) >= 2 and not is_generic and item in text:
            candidates[item] = 1.0
    query_emb = embeddings.embed_query(text)
    query_emb_np = np.array(query_emb, dtype=np.float32).reshape(1, -1)
    source_embeddings_np = np.array(risk_source_embeddings, dtype=np.float32)
    sims_matrix = cosine_similarity(query_emb_np, source_embeddings_np)
    sims = sims_matrix[0]
    for idx, score in enumerate(sims):
        item = risk_source_list[idx].strip()
        if score >= similarity_threshold and item not in candidates:
            candidates[item] = float(score)
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    final_results = [item for item, score in sorted_candidates]
    return final_results[:top_k]

# --- LangChain RAG Chains Definition ---

# 链 1: 图像描述 
image_describer_chain = (
    RunnableLambda(lambda input_dict: [
         SystemMessage(content="你是一个视觉描述助手，专注于精准识别图像中的工业核心设备和操作场景。"),
         HumanMessage(content=[
             {"type": "text", "text": """你的任务是：用一段简短的描述性文字，清晰、准确地总结图片中的核心实体和场景。

             **指令**:
             1.  **识别核心设备**: 请首先识别出图片中最主要、最关键的工业设备或机械，并明确说出它的具体名称（例如："一台大型工业切纸机"、"一台数控车床"）。
             2.  **描述周围环境**: 简要描述设备周围的环境、物料堆放情况以及任何可见的辅助工具。
             3.  **整合成段落**: 将所有信息组合成一个连贯的、单一段落的文本。
             4.  **保持客观**: 只描述你看到的客观事实，不要进行任何风险分析或提出建议。
             """},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_dict['image_base64']}"}}
         ])
    ])
    | llm
    | StrOutputParser()
)

# 链 2: LLM 视觉验证链
verifier_chain = (
    RunnableLambda(lambda input_dict: [
        SystemMessage(content="你是一个严谨的视觉分析助手，任务是从列表中筛选出与图片最相关的核心风险源。"),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f"""请仔细观察图片，并从下面的"候选风险源列表"中，选出图片中具体的**1-2个** 关键风险源。

                **候选风险源列表**:
                {input_dict['candidates']}

                **要求**:
                - **优先选择最具体、最核心的工业设备名称**（例如，在"机械设备"和"切纸机"中，优先选择"切纸机"）。
                - 只返回那些图片中真实存在的工业设备对应的风险源名称。
                - 注意保留与场景相关的风险源名称。
                - **严格按照此格式输出**: 将你选择的词汇用逗号(,)分隔，不要添加任何解释、编号或多余文字。
                - 如果列表中没有任何一项与图片相关，则返回空字符串。

                **示例输出**:
                切纸机,纸张
                """
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{input_dict['image_base64']}"}
            }
        ])
    ])
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda s: [item.strip() for item in s.split(',') if item.strip()])
)

# 添加文档过滤函数
def filter_docs_by_risk_sources(docs, verified_risk_sources, max_docs_per_source=6):
    """根据验证后的风险源过滤文档，并限制每个风险源的文档数量。"""
    if not verified_risk_sources:
        return []

    # 用于存储每个风险源已匹配的文档
    matched_docs_map = {source: [] for source in verified_risk_sources}

    for doc in docs:
        content = doc.page_content
        for risk_source in verified_risk_sources:
            # 如果当前风险源的文档数量已达到上限，则跳过
            if len(matched_docs_map[risk_source]) >= max_docs_per_source:
                continue

            # 检查文档内容是否与风险源相关
            if risk_source in content:
                matched_docs_map[risk_source].append(doc)
                # 一个文档只匹配给第一个找到的风险源，避免重复
                break 

    # 合并所有筛选出的文档，并去重
    final_docs = []
    seen_docs = set()
    for source in verified_risk_sources:
        for doc in matched_docs_map[source]:
            if doc.page_content not in seen_docs:
                final_docs.append(doc)
                seen_docs.add(doc.page_content)

    return final_docs


# 链 3: 检索链 - 修改为返回文档对象列表
retrieval_chain = (
    retriever
    | RunnableLambda(dedup_docs_simple)
)

# 链 4a: 风险识别与描述链（使用原有的llm）
risk_analysis_prompt = ChatPromptTemplate.from_template("""**任务**: 根据提供的材料，生成风险识别和描述报告。

**背景材料**:
1. **识别出的风险源**: {risk_sources}
2. **相关法规参考**: 
{context}

**你的指令**:
1.  **综合风险描述**: 按风险源分类列出相关的风险描述，格式如下：
   - 风险源名称1
     1. 风险描述内容1
     2. 风险描述内容2
   - 风险源名称2
     1. 风险描述内容1

2. **风险类型**: 根据风险描述内容，提炼出关键的风险类型，不超过5个，用逗号分隔。

**请严格按照以下格式输出:**

#### · 识别出的风险源
{risk_sources}

#### · 综合风险描述
[请根据法规参考重新整理，使用上述指定格式]

#### · 风险类型
[此处填写提炼的风险类型]

**完成上述内容后立即停止输出，不要生成管理措施。**
""")

risk_analysis_chain = (
    risk_analysis_prompt
    | llm  # 使用原有的llm
    | StrOutputParser()
)

# 链 4b: 管理措施生成链（使用限制token的management_llm）
management_measures_prompt = ChatPromptTemplate.from_template("""**任务**: 基于已识别的风险信息，生成针对性的综合管理措施。

**风险分析结果**:
{risk_analysis_result}

**你的指令**:
1.  针对每个风险源，提出1-2条最直接、最有效的管控措施。
2.  措施应具体、可操作，避免宽泛的建议。
3.  严格按照指定的格式输出，不要添加任何多余的解释或引言。

**输出格式**:

#### · 综合管理措施
- **针对[风险源名称]**：
  1. [具体措施1]
  2. [具体措施2]

**完成后立即停止输出。**
""")

management_measures_chain = (
    management_measures_prompt
    | management_llm  # 使用限制token的management_llm
    | StrOutputParser()
)

# --- FastAPI Application ---
app = FastAPI(title="本地Qwen2-VL风险源分析 RAG 系统")

class AnalysisResponse(BaseModel):
    description: str
    violations: str

@app.post("/analyze_image_local", response_model=AnalysisResponse)
async def analyze_image_local_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件类型错误，请上传图片文件。")

    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        print(f"收到图片: {file.filename}, 开始进行多步 RAG 分析...")

        # 步骤 1: 生成图像描述
        print("步骤 1: 生成图像描述...")
        description_result = await image_describer_chain.ainvoke({"image_base64": image_base64})
        print(f"生成描述 (部分): {description_result[:100]}...")
        cleanup_memory()

        # 【核心改动】步骤 1.5: 生成候选风险源列表
        print("步骤 1.5: 生成候选风险源列表...")
        candidate_risk_sources = get_hybrid_risk_sources(
            description_result,
            top_k=RISK_SOURCE_CANDIDATE_K,
            similarity_threshold=RISK_SOURCE_SIMILARITY_THRESHOLD
        )
        print(f"生成的候选列表: {candidate_risk_sources}")

        # 【核心改动】步骤 1.6: LLM 进行视觉验证
        print("步骤 1.6: LLM 视觉验证候选列表...")
        verified_risk_sources = []
        if candidate_risk_sources:
            verified_risk_sources = await verifier_chain.ainvoke({
                "image_base64": image_base64,
                "candidates": ", ".join(candidate_risk_sources)
            })
        print(f"LLM 验证后的风险源: {verified_risk_sources}")
        
        # 截取最终数量
        matched_risk_sources = verified_risk_sources[:RISK_SOURCE_TOP_K]
        print(f"最终选定的风险源: {matched_risk_sources}")
        cleanup_memory()

        # 步骤 2: 基于验证后的风险源检索法规...
        print("步骤 2: 基于验证后的风险源检索法规...")
        context_result = ""
        if matched_risk_sources:
            retrieval_query = f"请提供与以下风险源相关的法规、标准和安全操作规程：{'、'.join(matched_risk_sources)}"
            # 获取文档对象列表
            retrieved_docs = await retrieval_chain.ainvoke(retrieval_query)
            print(f"检索到 {len(retrieved_docs)} 篇原始文档。")

            # 步骤 2.5: 基于验证后的风险源过滤文档
            print("步骤 2.5: 过滤文档...")
            filtered_docs = filter_docs_by_risk_sources(retrieved_docs, matched_risk_sources, max_docs_per_source=6)
            print(f"过滤后剩余 {len(filtered_docs)} 篇相关文档。")

            # 新增步骤 2.6: 确保风险源有对应的文档，过滤掉没有文档的风险源
            final_risk_sources_with_docs = []
            if filtered_docs:
                # 从过滤后的文档中反向推断出哪些风险源是有效的
                all_content = "".join([doc.page_content for doc in filtered_docs])
                for rs in matched_risk_sources:
                    if rs in all_content:
                        final_risk_sources_with_docs.append(rs)
                print(f"确认有文档支持的风险源: {final_risk_sources_with_docs}")
            else:
                print("没有文档支持任何已识别的风险源。")

            if filtered_docs:
                context_result = format_docs(filtered_docs)
            else:
                context_result = "未找到与识别出的风险源直接相关的具体法规条文。"
        else:
            context_result = "未从图片中识别出明确的风险源，无法检索相关法规。"
            final_risk_sources_with_docs = [] # 确保在无风险源时列表为空
        cleanup_memory()

        # 在analyze_image_local_endpoint函数中，替换原有的步骤3
        
        # 步骤 3a: 生成风险识别与描述报告
        print("步骤 3a: 生成风险识别与描述报告...")
        risk_analysis_result = await risk_analysis_chain.ainvoke({
            "risk_sources": '、'.join(final_risk_sources_with_docs) if final_risk_sources_with_docs else '未识别',
            "context": context_result
        })
        print("风险识别与描述报告生成完毕。")
        cleanup_memory()
        
        # 步骤 3b: 生成综合管理措施（使用限制token的LLM）
        print("步骤 3b: 生成综合管理措施...")
        management_measures_result = await management_measures_chain.ainvoke({
            "risk_analysis_result": risk_analysis_result
        })
        print("综合管理措施生成完毕。")
        cleanup_memory()
        
        # 合并最终报告
        final_report = f"{risk_analysis_result}\n\n{management_measures_result}"
        return AnalysisResponse(description=description_result, violations=final_report)

    except Exception as e:
        print(f"处理图片时发生严重错误: {e}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # 允许从任何 IP 访问，端口为 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)