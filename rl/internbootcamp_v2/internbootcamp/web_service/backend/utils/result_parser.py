"""
结果解析器 - 用于解析评测结果并计算统计指标
"""
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class ResultParser:
    """评测结果解析器"""
    
    @staticmethod
    def parse_sample_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """解析单个样本结果 - 支持多种格式"""
        # 检测格式类型
        if "iterations" in result:
            # 新格式：包含完整迭代信息的格式
            return ResultParser._parse_iteration_format(result)
        else:
            # 旧格式：标准评测格式
            return ResultParser._parse_standard_format(result)
    
    @staticmethod
    def _parse_standard_format(result: Dict[str, Any]) -> Dict[str, Any]:
        """解析标准格式的评测结果"""
        input_data = result.get("input", {})
        extra_info = input_data.get("extra_info", {})
        reward_model = input_data.get("reward_model", {})
        turn_record = result.get("turn_record", {})
        
        # 计算 assistant turns 和 tool calls
        assistant_turns, tool_calls, interaction_turns = ResultParser._calculate_turn_stats(turn_record)
        
        token_usage = result.get("token_usage", {})
        
        return {
            "sample_id": extra_info.get("index", "unknown"),
            "data_source": input_data.get("data_source", "unknown"),
            "generator_name": extra_info.get("generator_name", ""),
            "score": result.get("score", 0.0),
            "success": result.get("success", False),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "assistant_turns": assistant_turns,
            "tool_calls": tool_calls,
            "interaction_turns": interaction_turns,
            "messages": result.get("messages", []),
            "extracted_output": result.get("extracted_output"),
            "ground_truth": reward_model.get("ground_truth"),
            "error": result.get("error"),
            "format_type": "standard",
        }
    
    @staticmethod
    def _parse_iteration_format(result: Dict[str, Any]) -> Dict[str, Any]:
        """解析包含迭代信息的新格式"""
        iterations = result.get("iterations", [])
        first_iteration = iterations[0] if iterations else {}
        input_data = first_iteration.get("input", {})
        
        # 提取基本信息
        data_source = input_data.get("data_source", "unknown")
        
        # 计算token统计
        total_response_tokens = result.get("total_response_tokens", 0)
        total_iteration_tokens = result.get("total_iteration_tokens", 0)
        
        # 统计所有迭代的turns和tool calls
        total_assistant_turns = 0
        total_tool_calls = 0
        for iteration in iterations:
            turn_record = iteration.get("turn_record", {})
            assistant_turns, tool_calls, _ = ResultParser._calculate_turn_stats(turn_record)
            total_assistant_turns += assistant_turns
            total_tool_calls += tool_calls
        
        # 计算最高分：取所有迭代中的最高分
        iteration_scores = result.get("iteration_scores", [])
        if iteration_scores:
            best_score = max(iteration_scores)
        else:
            best_score = result.get("final_score", 0.0)
        
        # 找到最高分对应的迭代索引
        best_iteration_idx = -1
        if iteration_scores:
            try:
                best_iteration_idx = iteration_scores.index(best_score)
            except ValueError:
                best_iteration_idx = -1
        
        return {
            "sample_id": "unknown",  # 新格式中没有明确的sample_id
            "data_source": data_source,
            "generator_name": "",
            "score": best_score,  # 使用最高分而不是final_score
            "success": result.get("success", False),
            "prompt_tokens": total_iteration_tokens - total_response_tokens,
            "completion_tokens": total_response_tokens,
            "total_tokens": total_iteration_tokens,
            "assistant_turns": total_assistant_turns,
            "tool_calls": total_tool_calls,
            "interaction_turns": len(iterations),
            "messages": [],  # 新格式中messages在iterations内部
            "extracted_output": result.get("extracted_output"),
            "ground_truth": result.get("ground_truth"),
            "error": result.get("error"),
            "format_type": "iteration",
            # 保留完整的迭代信息供前端展示
            "iterations": iterations,
            "iteration_scores": iteration_scores,
            "iterations_count": result.get("iterations_count", 0),
            "evaluation_config": result.get("evaluation_config", {}),
            # 添加最高分相关信息
            "best_score": best_score,
            "best_iteration_idx": best_iteration_idx,
            "final_score": result.get("final_score", 0.0),  # 保留最后一次的分数用于对比
        }
    
    @staticmethod
    def _calculate_turn_stats(turn_record: Dict[str, Any]) -> Tuple[int, int, int]:
        """计算轮次统计"""
        total_assistant_turns = 0
        total_tool_calls = 0
        total_interaction_turns = 0
        
        for turn_key, turn_data in turn_record.items():
            if turn_key.startswith("interaction_turn_"):
                total_assistant_turns += turn_data.get("assistant_turns", 0)
                total_tool_calls += turn_data.get("tool_calls_executed", 0)
                total_interaction_turns += 1
        
        return total_assistant_turns, total_tool_calls, total_interaction_turns
    
    @staticmethod
    def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算汇总统计"""
        if not results:
            return {
                "total_samples": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "avg_score": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "avg_tokens": 0.0,
            }
        
        success_results = [r for r in results if r.get("success", False)]
        
        total_samples = len(results)
        success_count = len(success_results)
        error_count = total_samples - success_count
        
        # 计算平均分（只统计成功的样本）
        if success_results:
            avg_score = sum(r.get("score", 0) for r in success_results if isinstance(r.get("score"), (int, float))) / len(success_results)
        else:
            avg_score = 0.0
        
        # 计算 token 统计
        total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in success_results)
        total_completion_tokens = sum(r.get("completion_tokens", 0) for r in success_results)
        total_tokens = sum(r.get("total_tokens", 0) for r in success_results)
        
        avg_prompt_tokens = total_prompt_tokens / success_count if success_count > 0 else 0.0
        avg_completion_tokens = total_completion_tokens / success_count if success_count > 0 else 0.0
        avg_tokens = total_tokens / success_count if success_count > 0 else 0.0
        
        return {
            "total_samples": total_samples,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / total_samples if total_samples > 0 else 0.0,
            "avg_score": avg_score,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_tokens": avg_tokens,
        }
    
    @staticmethod
    def calculate_data_source_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """按数据源分组统计"""
        data_source_groups = {}
        
        for result in results:
            data_source = result.get("data_source", "unknown")
            if data_source not in data_source_groups:
                data_source_groups[data_source] = []
            data_source_groups[data_source].append(result)
        
        stats = {}
        for data_source, group_results in data_source_groups.items():
            success_results = [r for r in group_results if r.get("success", False)]
            
            scores = [r.get("score", 0) for r in success_results if isinstance(r.get("score"), (int, float))]
            
            total_count = len(group_results)
            success_count = len(success_results)
            
            stats[data_source] = {
                "data_source": data_source,
                "total_count": total_count,
                "success_count": success_count,
                "error_count": total_count - success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "avg_assistant_turns": sum(r.get("assistant_turns", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                "avg_tool_calls": sum(r.get("tool_calls", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                "avg_interaction_turns": sum(r.get("interaction_turns", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                "avg_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                "avg_completion_tokens": sum(r.get("completion_tokens", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                "avg_tokens": sum(r.get("total_tokens", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
            }
        
        return stats
    
    @staticmethod
    def calculate_generator_stats(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按生成器分组统计（分组到数据源下）"""
        # 首先按数据源分组
        data_source_groups = {}
        for result in results:
            data_source = result.get("data_source", "unknown")
            if data_source not in data_source_groups:
                data_source_groups[data_source] = {}
            
            generator_name = result.get("generator_name", "")
            if generator_name:
                if generator_name not in data_source_groups[data_source]:
                    data_source_groups[data_source][generator_name] = []
                data_source_groups[data_source][generator_name].append(result)
        
        # 计算每个生成器的统计
        all_stats = {}
        for data_source, generators in data_source_groups.items():
            all_stats[data_source] = []
            
            for generator_name, group_results in generators.items():
                success_results = [r for r in group_results if r.get("success", False)]
                scores = [r.get("score", 0) for r in success_results if isinstance(r.get("score"), (int, float))]
                
                total_count = len(group_results)
                success_count = len(success_results)
                
                all_stats[data_source].append({
                    "generator_name": generator_name,
                    "total_count": total_count,
                    "success_count": success_count,
                    "error_count": total_count - success_count,
                    "success_rate": success_count / total_count if total_count > 0 else 0.0,
                    "avg_score": sum(scores) / len(scores) if scores else 0.0,
                    "max_score": max(scores) if scores else 0.0,
                    "min_score": min(scores) if scores else 0.0,
                    "avg_assistant_turns": sum(r.get("assistant_turns", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                    "avg_tool_calls": sum(r.get("tool_calls", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                    "avg_interaction_turns": sum(r.get("interaction_turns", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                    "avg_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                    "avg_completion_tokens": sum(r.get("completion_tokens", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                    "avg_tokens": sum(r.get("total_tokens", 0) for r in success_results) / success_count if success_count > 0 else 0.0,
                })
        
        return all_stats
    
    @staticmethod
    def extract_errors(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取错误信息"""
        errors = []
        for result in results:
            if not result.get("success", False):
                errors.append({
                    "data_source": result.get("data_source", "unknown"),
                    "generator_name": result.get("generator_name", ""),
                    "error": result.get("error", "Unknown error"),
                    "sample_id": str(result.get("sample_id", "unknown")),
                })
        return errors

