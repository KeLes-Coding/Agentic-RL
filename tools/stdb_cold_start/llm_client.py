"""
LLM Client - OpenAI SDK兼容的LLM客户端封装
支持DeepSeek及其他OpenAI兼容API
"""
import re
import time
import logging
from typing import Tuple, Optional
from openai import OpenAI

from .config import ColdStartConfig

logger = logging.getLogger(__name__)


# ALFWORLD Prompt模板
ALFWORLD_SYSTEM_PROMPT = """You are an expert agent operating in the ALFRED Embodied Environment. 
Your goal is to complete household tasks by taking actions step by step.

IMPORTANT RULES:
1. Always choose actions from the provided admissible actions list
2. Think step by step about the current situation before acting
3. Use <think>...</think> tags for your reasoning
4. Use <action>...</action> tags for your chosen action
5. The action must be EXACTLY one of the admissible actions (copy it exactly)"""

ALFWORLD_USER_TEMPLATE = """Your task is: {task_description}

You have taken {step_count} step(s) so far.
{history_section}

Current observation: {current_observation}

Admissible actions: [{admissible_actions}]

Think about the current situation and choose ONE action from the admissible actions list.
Remember to use <think>...</think> for reasoning and <action>...</action> for the action."""


class LLMClient:
    """OpenAI SDK兼容的LLM客户端"""
    
    def __init__(self, config: ColdStartConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.request_timeout
        )
        self.model = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        
        # 统计
        self.total_tokens_used = 0
        self.total_requests = 0
    
    def generate_action(
        self,
        task_description: str,
        current_observation: str,
        admissible_actions: list,
        step_count: int = 0,
        action_history: Optional[list] = None
    ) -> Tuple[str, str, str, int]:
        """
        生成动作
        
        Args:
            task_description: 任务描述
            current_observation: 当前观测
            admissible_actions: 可选动作列表
            step_count: 当前步数
            action_history: 历史动作列表 [(obs, action), ...]
        
        Returns:
            Tuple of (think_content, action_raw, llm_response, tokens_used)
        """
        # 构建history部分
        history_section = ""
        if action_history and len(action_history) > 0:
            recent_history = action_history[-5:]  # 最近5步
            history_lines = []
            for i, (obs, act) in enumerate(recent_history):
                obs_short = obs[:200] + "..." if len(obs) > 200 else obs
                history_lines.append(f"Step {step_count - len(recent_history) + i + 1}: Action: {act}\n  Observation: {obs_short}")
            history_section = "Recent history:\n" + "\n".join(history_lines)
        else:
            history_section = "This is your first step."
        
        # 构建用户消息
        user_message = ALFWORLD_USER_TEMPLATE.format(
            task_description=task_description,
            step_count=step_count,
            history_section=history_section,
            current_observation=current_observation,
            admissible_actions=", ".join(admissible_actions)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            llm_response = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            self.total_tokens_used += tokens_used
            self.total_requests += 1
            
            # 解析响应
            think_content = self._extract_tag_content(llm_response, "think")
            action_raw = self._extract_tag_content(llm_response, "action")
            
            return think_content, action_raw, llm_response, tokens_used
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _extract_tag_content(self, text: str, tag: str) -> str:
        """从文本中提取指定标签的内容"""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def match_action(self, action_raw: str, admissible_actions: list) -> Optional[str]:
        """
        将LLM输出的动作匹配到admissible_actions
        
        Returns:
            匹配的动作，如果无法匹配返回None
        """
        if not action_raw:
            return None
        
        action_lower = action_raw.lower().strip()
        
        # 精确匹配
        for act in admissible_actions:
            if act.lower().strip() == action_lower:
                return act
        
        # 包含匹配
        for act in admissible_actions:
            if action_lower in act.lower() or act.lower() in action_lower:
                return act
        
        # 模糊匹配（去除数字后缀）
        action_base = re.sub(r'\s*\d+\s*$', '', action_lower)
        for act in admissible_actions:
            act_base = re.sub(r'\s*\d+\s*$', '', act.lower())
            if action_base == act_base:
                return act
        
        return None
