import os
import base64
import logging
from typing import List, Optional
from io import BytesIO
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from PIL import Image
from openai import OpenAI

# ==========================================
# 1. 初始化工业级流式日志
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger("VLM_OCR_Parser")

# 加载高级环境变量范式
load_dotenv()

# ==========================================
# 2. 结构化契约定义 (Data Schema)
# ==========================================
class TableRow(BaseModel):
    item_name: str = Field(description="条目/商品名称或费用项名称")
    specification: Optional[str] = Field(None, description="规格型号/单位/数量，若无则留空")
    amount: float = Field(description="金额/数值 (不含货币符号，纯数字)")

class StructuredInvoice(BaseModel):
    """定义发票/报表高精度抽取的最终结构化契约"""
    invoice_title: str = Field(description="发票或报表的标题/主体名称")
    invoice_number: Optional[str] = Field(None, description="发票号码/代码/报表批次号")
    date: Optional[str] = Field(None, description="开票日期或报表日期，统一转换为 YYYY-MM-DD 格式")
    line_items: List[TableRow] = Field(default=[], description="表格中的多行细目数据")
    total_amount: float = Field(description="合计总金额/总计数值")
    audit_remark: str = Field(description="多模态眼睛对该图的合规性审计评语（如是否模糊、是否有盖章错位、是否有涂改风险等）")

# ==========================================
# 3. 核心生产级加工类定义
# ==========================================
class MultimodalDataPipeline:
    def __init__(self):
        # 从环境变量安全获取，兼容 deepseek-v4 等多模态网关
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        
        if not self.api_key:
            raise ValueError("🚨 核心硬伤：未在 .env 文件中检测到 DEEPSEEK_API_KEY")
            
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info("✅ 多模态数据流客户端初始化成功。")

    def _optimize_image(self, image_path: str, max_size: int = 1024, quality: int = 80) -> bytes:
        """
        工业级前置图像优化引擎
        防御性：防止过大、分辨率超标的图片直接打入模型，导致高昂 Token 消耗或网关拒绝服务
        """
        logger.info(f"⚙️ 启动客户端图像清洗: {image_path}")
        try:
            with Image.open(image_path) as img:
                # 兼容不同图片模式（如 RGBA 转为 RGB，避免 JPEG 格式不支持透明度而报错）
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # 动态比例缩放控制 (Bounding Box 缩放算法)
                width, height = img.size
                if max(width, height) > max_size:
                    if width > height:
                        new_width = max_size
                        new_height = int(height * (max_size / width))
                    else:
                        new_height = max_size
                        new_width = int(width * (max_size / height))
                    
                    logger.warning(f"⚠️ 图片分辨率过高 ({width}x{height})，实施断崖式等比限流缩放至: {new_width}x{new_height}")
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 压缩并输出内存流
                output_buffer = BytesIO()
                img.save(output_buffer, format="JPEG", quality=quality)
                compressed_data = output_buffer.getvalue()
                
                logger.info(f"🚀 图像清洗完成。原始大小: {os.path.getsize(image_path)/1024:.2f}KB, 压缩后大小: {len(compressed_data)/1024:.2f}KB")
                return compressed_data
                
        except Exception as e:
            logger.error(f"🚨 客户端图片处理崩溃: {str(e)}")
            raise

    def parse_invoice_image(self, image_path: str) -> StructuredInvoice:
        """
        执行多模态结构化高精度提取
        """
        # 1. 先进行图片防御性重构
        optimized_img_bytes = self._optimize_image(image_path)
        
        # 2. 进行安全的客户端 Base64 编码
        base64_image = base64.b64encode(optimized_img_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{base64_image}"
        
        # 3. 构造最高安全级别的多模态统一系统提示词
        system_prompt = (
            "你是一个工业级的多模态高精度视觉 OCR 解析器。\n"
            "你的任务是仔细审视用户提供的图片（包含发票、表格或财务报表），将其中混乱的视觉信号转化为高可信度的结构化输出。\n"
            "请严格按照指定的 JSON Schema 返回结果。如果遇到图片中的手写数字或倾斜表格，请发挥多模态空间感知直觉进行对齐。"
        )
        
        # 4. 发起 API 请求 (严格对齐标准多模态 Payload 规范)
        logger.info("📡 正在向 VLM 大模型网关推送多模态报文...")
        try:
            # 采用具备高性价比、多模态支持的最新 Flash 模型 (如 deepseek-v4-flash)
            # 在实际生产中，由于使用了 response_format，框架将自动绑定强校验
            response = self.client.beta.chat.completions.parse(
                model="deepseek-v4-flash", # 或使用你的具体可用视觉模型名称
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "请提取这张财务报表/发票截图中的关键信息，并对排版合规性做出审计。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ],
                response_format=StructuredInvoice, # 强行挂载 Pydantic 契约
                temperature=0.0 # 严禁多模态进行自由创作，将温度拉至 0
            )
            
            logger.info("✅ 接收到大模型多模态结构化报文。")
            return response.choices[0].message.parsed
            
        except Exception as e:
            logger.error(f"🚨 多模态 API 通信或解析失败: {str(e)}")
            # 这里留出 Safe Fallback（安全兜底机制）位置
            raise

# ==========================================
# 4. 本地 MVP 最小可行性冒烟测试
# ==========================================
if __name__ == "__main__":
    # 模拟一张本地测试图片（测试前请确保在当前目录下放一张真正的图片，或修改此处的路径）
    mock_image_path = "multi_model/data/fp1.webp"
    
    # 工业级防御：如果用户忘记放图片，先生成一张假的测试图，防止脚本直接崩溃
    if not os.path.exists(mock_image_path):
        logger.warning(f"未在本地发现测试图片 {mock_image_path}，正在自动为您生成一张临时画布进行冒烟测试...")
        try:
            from PIL import ImageDraw
            img = Image.new('RGB', (1200, 800), color='#faf8f5')
            d = ImageDraw.Draw(img)
            d.text((50, 50), "Mock Invoice \nNo: 20260623\nTotal: 1250.00\nItem A: 500.00\nItem B: 750.00", fill='#000000')
            img.save(mock_image_path)
            logger.info(f"🎨 临时测试画布 {mock_image_path} 创建完成。")
        except ImportError:
            logger.error("🚨 无法创建测试画布，请在本地放置真实的 invoice_demo.jpg 图像文件。")
            exit(1)

    # 启动解析引擎
    try:
        parser_engine = MultimodalDataPipeline()
        structured_result = parser_engine.parse_invoice_image(mock_image_path)
        
        print("\n" + "="*50)
        print("🎉 工业级多模态 OCR 结构化提取成功 🎉")
        print("="*50)
        print(structured_result.model_dump_json(indent=2))
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试流程终止，系统捕获到非故障崩溃隐患: {e}")