import os
from typing import Dict, List, Optional
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage
from langchain_core.callbacks import StdOutCallbackHandler
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

RAG_PROMPT_TEMPLATE="""
# 角色与任务
你是一名专业的铁路与交通标准分析师，擅长基于提供的技术资料进行深度分析和推理。你的核心任务是：**基于资料内容进行合理的逻辑推断和综合分析**。

# 可用资料
{context}

# 待分析问题  
{question}

# 分析原则
1. **主动推理**：不仅要找出直接相关的内容，还要基于多个信息点进行逻辑连接和合理推断
2. **证据导向**：所有结论必须基于资料中的证据链，可以是通过多个段落综合推理得出
3. **层次分析**：对于复杂问题，先分析各个组成部分，再综合得出结论
4. **合理推测**：当资料提供足够线索时，可以进行基于证据的合理推测，并明确标注为"推断"

# 期望的分析深度
- **事实问题**：直接定位并引用具体信息
- **比较问题**：找出对比项的共同点和差异，分析背后的原因
- **因果问题**：构建因果关系链，解释"为什么"
- **流程问题**：梳理步骤逻辑和时间顺序
- **数值分析**：如有多个相关数值，分析其关系和趋势

# 输出要求
请以分析报告的形式回答，包含以下部分：

## 主要结论
[基于资料推理得出的核心答案，即使需要连接不同信息点也要给出明确结论]

## 分析过程
- 列出用于推理的关键信息点及其来源
- 说明信息点之间的逻辑关系
- 展示从信息到结论的推理路径

## 证据引用
对所有引用的资料标注具体来源：`【文档名称】`

## 确定性说明
- 高确定性：资料提供直接且充分的证据
- 中等确定性：基于多个信息点合理推断  
- 低确定性：资料仅提供部分线索，结论存在不确定性

**重要**：只有在资料完全没有任何相关信息时才说"无法回答"。只要有任何相关线索，就要尝试进行分析和推理。
"""


class BaseModel:
    def __init__(self, model) -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass


class OpenAIChat(BaseModel):
    def __init__(self, model: str = "deepseek", default_temperature: Optional[float] = None) -> None:
        self.model = model
        # 允许通过环境变量设置默认温度，便于在不同问题类型间复用
        if default_temperature is None:
            try:
                default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
            except ValueError:
                default_temperature = 0.7
        self.default_temperature = default_temperature

    def chat(
        self,
        prompt: str,
        history: List[dict],
        content: str,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        llm = ChatOpenAI(
            model_name=os.getenv("CLOUD_MODEL"),
            openai_api_key=os.getenv("CLOUD_API_KEY"),
            openai_api_base=os.getenv("CLOUD_BASE_URL"),
            callbacks=[StdOutCallbackHandler()],  # 实时打印生成内容
            temperature=self.default_temperature if temperature is None else temperature
        )
        final_template = prompt_template or RAG_PROMPT_TEMPLATE

        history.append({'role': 'user', 'content': final_template.format(question=prompt, context=content)})

        response = llm.invoke([HumanMessage(content=message['content']) for message in history])

        return response.content


if __name__ == "__main__":
    model = OpenAIChat()
    response = model.chat("中国的首都是哪里？", [], "中国的首都是北京。")
    print(response)
