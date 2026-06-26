import re
import math
import json
from typing import Optional, Dict, Any
from internbootcamp.src.base_reward_calculator import BaseRewardCalculator
import random
from .math_reward import compute_score
import requests
from .judge_model_infos import judge_model_infos
from statistics import mean  

def call_judge_model(prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.7):
    """
    使用 requests 调用 Judge 模型
    """
    # 1. 拼接完整的 API Endpoint
    #以此配置为例，通常是 http://IP:PORT/v1/chat/completions
    base_url = judge_model_infos['model_url'].rstrip('/')
    api_url = f"{base_url}/chat/completions"

    # 2. 设置请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {judge_model_infos['model_api_key']}"
    }

    # 3. 构造请求体 (Payload)
    payload = {
        "model": judge_model_infos['model_name'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048,  #根据需要调整
        "stream": False
    }

    try:
        # 4. 发送 POST 请求
        print(f"正在请求: {api_url} ...")
        response = requests.post(
            api_url, 
            headers=headers, 
            json=payload, 
            timeout=60 # 设置超时时间防止卡死
        )

        # 5. 检查响应状态码
        response.raise_for_status()

        # 6. 解析结果
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        return content

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        if 'response' in locals():
            print(f"服务器返回内容: {response.text}")
        return None

class LlmTimerRewardCalculator(BaseRewardCalculator):
    """示例奖励管理器，用于评估算术运算任务"""
    
    @staticmethod
    def extract_output(output_str: str) -> Optional[Dict[str, Any]]:
        """
        从模型输出中提取算术运算结果
        
        Args:
            output_str (str): 模型的原始输出
            
        Returns:
            Optional[Dict[str, Any]]: 提取的运算信息，包含操作符、操作数和结果
        """
        
        # 1. EXTRACT REASONING ANSWER
        # 1.1 extract for general reasoning : 
        #    有些模型会在中间复述指令或示例，也会带上 <answer>...</answer>，
        #    为了拿到真正“最终回答”，这里取最后一个匹配而不是第一个。
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        answer_matches = answer_pattern.findall(output_str)
        if answer_matches:
            reasoning_answer = answer_matches[-1].strip()
        else:
            reasoning_answer = None
        
        # 1.2 extract for final time conclusion :
        time_conclusion_pattern = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)
        time_conclusion_match = time_conclusion_pattern.search(output_str)
        if time_conclusion_match:
            time_conclusion = time_conclusion_match.group(1).strip()
        else:
            time_conclusion = None
        
        # 1.3 extract final time call (tool response) :
        time_call_pattern = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
        all_matches = time_call_pattern.findall(output_str)
        
        if all_matches:
            # 取列表中的最后一个，并去除首尾空格
            time_call = all_matches[-1].strip()
        else:
            time_call = None

        # 2. EXTRACT JERICHO TOOL CALLS
        All_tool_calls = all_matches
        time_conclusion_pattern = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)
        score_conclusion_pattern = re.compile(r"<score>(.*?)</score>", re.DOTALL)
        
        # 使用 findall 获取所有匹配的内容列表
        time_matches = time_conclusion_pattern.findall(output_str)
        score_matches = score_conclusion_pattern.findall(output_str)
        
        # 取列表的最后一个 [-1]
        if time_matches:
            time_conclusion = time_matches[-1].strip()
        else:
            time_conclusion = None
            
        if score_matches:
            score_conclusion = score_matches[-1].strip()
        else:
            score_conclusion = None

        # 3. EXTRACT MACHINE LEARNING INFOS
        All_tool_calls = all_matches
        time_conclusion_pattern = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)
        accuracy_conclusion_pattern = re.compile(r"<accuracy>(.*?)</accuracy>", re.DOTALL)

        time_matches = time_conclusion_pattern.findall(output_str)
        accuracy_matches = accuracy_conclusion_pattern.findall(output_str)

        if time_matches:
            time_conclusion = time_matches[-1].strip()
        else:
            time_conclusion = None
        
        if accuracy_matches:
            accuracy_conclusion = accuracy_matches[-1].strip()
        else:
            accuracy_conclusion = None
            
        return {
            "reasoning_RM_extracted_infos": {
                "reasoning_answer": reasoning_answer, 
                "time_conclusion": time_conclusion,
                "time_call": time_call
            }, 
            "jericho_RM_extracted_infos": {
                "all_tool_calls": All_tool_calls,
                "time_conclusion": time_conclusion,
                "score_conclusion": score_conclusion
            }, 
            "machine_learning_RM_extracted_infos": {
                "all_tool_calls": All_tool_calls,
                "time_conclusion": time_conclusion,
                "accuracy_conclusion": accuracy_conclusion
            }
        }

    @classmethod
    def _verify_correction(cls, extracted_output, identity: dict, **kwargs) -> float:
        """
        验证用户输出并计算得分
        
        Args:
            extracted_output: 提取的运算信息
            identity (dict): 任务信息，包含期望的运算和结果
            score_max (float): 最高分数
            score_min (float): 最低分数
            
        Returns:
            float: 得分
        """
        
        if identity.get("data_type") == "general_reasoning": # 1. general reasoing 
            
            reasoning_answer = extracted_output.get("reasoning_RM_extracted_infos").get("reasoning_answer")
            time_conclusion = extracted_output.get("reasoning_RM_extracted_infos").get("time_conclusion")
            last_time_call = extracted_output.get("reasoning_RM_extracted_infos").get("time_call")
            
            score = 0.0
            # 1. format score 0.1
            if reasoning_answer != None : 
                score += 0.05
            if time_conclusion != None : 
                score += 0.025
            if last_time_call != None : 
                score += 0.025
            
            Format_ERR = False
            if reasoning_answer is None or time_conclusion is None or last_time_call is None: # 如果任意一项为空，则直接返回format score
                if reasoning_answer is None:
                    print("[Debug LLMTimer Reward Calculator] reasoning answer is None.")
                    return score
                if time_conclusion is None:
                    print("[Debug LLMTimer Reward Calculator] time conclusion is None.")
                if last_time_call is None:
                    print("[Debug LLMTimer Reward Calculator] last time call is None.")
                print("[Debug LLMTimer Reward Calculator] response format error.")
                Format_ERR = True
            
            if Format_ERR:
                return score

            # 2. content score

            # 2.1 reasoning answer score
            print(f"reasoning answer: {reasoning_answer[:100]}, identity answer: {identity.get('answer')}")
            accuracy_score = compute_score(reasoning_answer, identity.get("answer"))

            if accuracy_score != 1.0: 
                print("[Debug LLMTimer Reward Calculator] Rule based answer correctness is [False], call judge model to further judge.")
                judge_model_system_prompt = "You are a helpful assistant. You are given two answers, the first is given by a model, and the second is the correct answer. You need to judge if the model's answer is correct. If the model's answer is correct, return \"yes\", otherwise return \"no\". Do not return any other text."

                question = identity.get("question", None)
                if question is not None:
                    judge_model_prompt = f"Question: {question}\nModel answer: {reasoning_answer}, Correct answer: {identity.get('answer')}"
                else:
                    judge_model_prompt = f"Model answer: {reasoning_answer}, Correct answer: {identity.get('answer')}"

                content_score = call_judge_model(judge_model_prompt, system_prompt=judge_model_system_prompt, temperature=0.0)
                if content_score == "yes" or content_score == "Yes" or content_score == "YES":
                    accuracy_score = 1.0
                else:
                    accuracy_score = 0.0
                print(f"[Debug LLMTimer Reward Calculator] Judge model result is: [{accuracy_score}]")
            
            if Format_ERR:
                return score + 0.4 * accuracy_score

            # 2.2 time conclusion score

            time_conclusion_pattern = r"total duration:\s*(\d+(?:\.\d+)?)\s*seconds"
            match = re.search(time_conclusion_pattern, time_conclusion, re.IGNORECASE) # re.IGNORECASE 让它不区分大小写
            
            if match:
                number_str = match.group(1)
                total_duration = float(number_str)
            else:
                total_duration = None  
            
            last_time_call_pattern = r"(\d+(?:\.\d+)?)\s*seconds"
            
            match = re.search(last_time_call_pattern, last_time_call, re.IGNORECASE)
            
            if match:
                last_time_call_duration = float(match.group(1))
            else:
                last_time_call_duration = None
            
            required_time = float(identity.get("required_time"))

            # Checks for 一些时间的声明
            if not last_time_call_duration or not total_duration:
                print("[Debug LLMTimer Reward Calculator] time conclusion or last time call duration is None.")
                return score + 0.4 * accuracy_score
            
            if not last_time_call_duration == total_duration: # 如果模型总结的最终时间不正确
                print("[Debug LLMTimer Reward Calculator] time conclusion or last time call duration is not equal.")
                print(f"total duration: {total_duration}, last time call duration: {last_time_call_duration}")
                return score + 0.4 * accuracy_score
            
            if identity.get("with_time_limit", True): # 当需要时间限制时
                # === 修改点 1: 调整基础时间奖励 ===
                if last_time_call_duration <= (1.0 + 1e-3) * required_time:
                    # 给 0.1 的基础合规分
                    # 剩下的 0.1 放到后面根据利用率动态给
                    score += 0.2 
                else :
                    # 超时处理：计算超时惩罚
                    print("[Debug LLMTimer Reward Calculator] last time call duration is greater than required time.")
                    overtime_ratio = (last_time_call_duration - required_time) / required_time
                    # 超时惩罚：最多扣0.2分，按超时比例线性增长
                    overtime_penalty = min(0.2, 0.2 * overtime_ratio)
                    print(f"[Debug] Overtime ratio: {overtime_ratio:.2f}, Penalty: {overtime_penalty:.4f}")
                    
                    score += 0.1 # format score
                    score += 0.5 * accuracy_score # 还是给点准确分
                    score -= overtime_penalty # 减去超时惩罚
                    score = max(0.0, score) # 确保分数不为负
                    return score

                # 2.1 reasoning answer score
                print("[Debug LLMTimer Reward Calculator] All checks passed, compute full score.")
                
                # === 修改点 2: 分段式时间利用率计算 ===
                
                # 计算时间利用率 (0.0 ~ 1.0)
                time_ratio = min(last_time_call_duration / required_time, 1.0)
                
                # 分段函数计算时间利用率分数
                if time_ratio < 0.25:
                    time_utilization_score = 0.025
                elif time_ratio < 0.5:
                    time_utilization_score = 0.05
                elif time_ratio < 0.75:
                    time_utilization_score = 0.075
                else:  # time_ratio >= 0.75
                    time_utilization_score = 0.1
                
                print(f"[Debug] Time Ratio: {time_ratio:.2f}, Utilization Score: {time_utilization_score:.4f}")
                
                # 加上动态时间分
                score += time_utilization_score
                
                # 加上准确率分 (最大 0.6)
                score += 0.6 * accuracy_score 
                
                # 最终总分 Max = 0.1(格式) + 0.1(合规) + 0.1(利用率) + 0.7(准确) = 1.0
            
            else :
                print("[Debug LLMTimer Reward Calculator] without time limit, return score.")
                score += 0.9 * accuracy_score
            return score

        elif identity.get("data_type") == "jericho": # 2. jericho
            
            all_tool_calls = extracted_output.get("jericho_RM_extracted_infos").get("all_tool_calls")
            time_conclusion = extracted_output.get("jericho_RM_extracted_infos").get("time_conclusion")
            score_conclusion = extracted_output.get("jericho_RM_extracted_infos").get("score_conclusion")
            
            # print(f"all_tool_calls: {all_tool_calls}, time_conclusion: {time_conclusion}, score_conclusion: {score_conclusion}")
            score = 0.0
            # 1. format score
            if all_tool_calls != None : 
                score += 0.05
            if time_conclusion != None : 
                score += 0.025
            if score_conclusion != None : 
                score += 0.025
            
            if time_conclusion is None or score_conclusion is None: # 如果任意一项为空，则直接返回format score
                if identity.get("with_time_limit", True):
                    print("[Debug LLMTimer Reward Calculator] response format error.")
                    return score
            
            if all_tool_calls is None: 
                print("[Debug LLMTimer Reward Calculator] response format error for NO time limit responses.")
                return score

            def extract_scores(str_list):
                scores = []
                pattern = re.compile(r'(?:Your current score is|Final score):\s*(\d+)')
                for text in str_list:
                    matches = pattern.findall(text)
                    for num in matches:
                        scores.append(float(num))
                        
                return scores
            
            # 2. score and time match score

            # 2.1 score conclusion match 0.05
            matches = []
            step_rewards = []
            time_duration_pattern = re.compile(r'You have played for (\d+(?:\.\d+)?) seconds\.')
            step_reward_pattern = re.compile(r'The step reward is: (\d+)')
            
            for tool_call in all_tool_calls:
                match = time_duration_pattern.search(tool_call)
                if match:
                    matches.append(float(match.group(1)))
                reward_match = step_reward_pattern.search(tool_call)
                if reward_match:
                    step_rewards.append(float(reward_match.group(1)))
            
            scores = extract_scores(all_tool_calls)
            if identity.get("with_time_limit", True):
                print(f"[Debug LLMTimer Reward Calculator] scores: {scores}, score_conclusion: {score_conclusion}")
                if float(score_conclusion) in scores or float(sum(step_rewards)) == float(score_conclusion): # 诚实汇报分数
                    score += 0.05
                else :
                    print("[Debug LLMTimer Reward Calculator] score conclusion is not honest.")
                    if len(scores) == 0:
                        print("[Debug LLMTimer Reward Calculator] score conclusion is not honest, and no score occurs.")
                        return score
                    else :
                        score_conclusion = max(scores)
                    
            
            # 当且仅当需要时间限制时，才计算时间相关得分
            if identity.get("with_time_limit", True):
                # 2.2 time conclusion match 0.05
                if len(matches) == 0:
                    print("[Debug LLMTimer Reward Calculator] time duration is not found, return score.")
                    return score
                else : 
                    time_conclusion_pattern = r"total duration:\s*(\d+(?:\.\d+)?)\s*seconds"
                    time_conclusion_match = re.search(time_conclusion_pattern, time_conclusion, re.IGNORECASE) # re.IGNORECASE 让它不区分大小写
                
                    if time_conclusion_match:
                        number_str = time_conclusion_match.group(1)
                        total_duration = float(number_str)
                    else:
                        total_duration = None  
                    
                    if total_duration == None:
                        print("[Debug LLMTimer Reward Calculator] time duration is not found, return score.")
                        return score

                    # print(matches)

                    if not total_duration == matches[-1]: # 如果模型总结的时间不正确
                        print(f"total duration: {total_duration}, last time call duration: {matches[-1]}")
                        print("[Debug LLMTimer Reward Calculator] time duration is not equal to time conclusion, return score.")
                        total_duration = max(total_duration, matches[-1]) # 选取一个最大的作为duration
                        # return score
                    else :
                        score += 0.05
                
                # 2.3 ON TIME SCORE 0.1 + 0.2 * on time curve
                required_time = float(identity.get("required_time"))
                max_score = identity.get("max_score")
                if total_duration <= (1.0 + 1e-3) * required_time :
                    # 分段式时间利用率计算
                    time_ratio = min(total_duration / required_time, 1.0)
                    
                    # 分段函数计算时间利用率分数
                    if time_ratio < 0.25:
                        time_utilization_score = 0.025
                    elif time_ratio < 0.5:
                        time_utilization_score = 0.05
                    elif time_ratio < 0.75:
                        time_utilization_score = 0.075
                    else:  # time_ratio >= 0.75
                        time_utilization_score = 0.1
                    
                    score += time_utilization_score
                    
                else :
                    print("[Debug LLMTimer Reward Calculator] total duration is greater than required time.")
                    score += 0.1 * ((float(score_conclusion) / float(max_score)) ** 0.5)
                    return score

            max_score = identity.get("max_score")
            # 3. 模型探索reward赋分
            if identity.get("with_time_limit", True):
                max_reward_score = 0.7
                exploration_reward_score = max_reward_score * ((float(score_conclusion) / float(max_score)) ** (1.0/3.0)) # 探索奖励分数, 最大为0.8
                print(f"[Debug LLMTimer Reward Calculator] current score: {score}, exploration reward score: {exploration_reward_score}, Final score: {score + exploration_reward_score}, max_score: {max_score}, score_conclusion: {score_conclusion}")
                score += exploration_reward_score
                return score
            else :
                # print(score_conclusion)
                # print(scores)
                if score_conclusion is None :
                    if len(scores) == 0:
                        print("[Debug LLMTimer Reward Calculator] scores is empty, return score.")
                        return score
                    score_conclusion = max(scores)
                elif not score_conclusion in scores:
                    if len(scores) == 0:
                        print("[Debug LLMTimer Reward Calculator] scores is empty, return score.")
                        return score
                    score_conclusion = max(scores)

                else :
                    score_conclusion = float(score_conclusion)

                max_reward_score = (1.0-0.1-0.05)

                # 0.1 for format score, 0.05 for score conclusion score
                exploration_reward_score = max_reward_score * ((float(score_conclusion) / float(max_score)) ** (1.0/3.0)) # 探索奖励分数, 最大为0.8
                print(f"[Debug LLMTimer Reward Calculator] current score: {score}, exploration reward score: {exploration_reward_score}, Final score: {score + exploration_reward_score}, max_score: {max_score}, score_conclusion: {score_conclusion}")
                score += exploration_reward_score
                return score
            return score

        elif identity.get("data_type") == "MachineLearning": # 3. ml
            if identity.get("single_round", False): # 默认是False，表示是多轮任务
                print("[Debug LLMTimer Reward Calculator] single round, computing single round ACC score.")
                ACCURACY_PATTERN = r"Evaluation accuracy: (\d+\.\d+)"
                extracted_accuracies = []
                all_tool_responses = extracted_output.get("machine_learning_RM_extracted_infos").get("all_tool_calls")

                for response in all_tool_responses:
                    match = re.search(ACCURACY_PATTERN, response)
                    if match:
                        accuracy_str = match.group(1)
                        accuracy_float = float(accuracy_str)
                        extracted_accuracies.append(accuracy_float)

                score = 0.0
                if len(extracted_accuracies) == 0:
                    print("[Debug LLMTimer Reward Calculator] no accuracy found, return score.")
                    return score
                else:
                    score += 0.2 # format score
                
                score += 0.8 * max(extracted_accuracies)
                print(f"[Debug LLMTimer Reward Calculator] score: {score}, extracted best accuracy: {max(extracted_accuracies)}")

                return score
            
            else : #也就是说是多轮任务
                ACCURACY_PATTERN = r"Evaluation accuracy: (\d+\.\d+)"
                extracted_accuracies = []

                all_tool_responses = extracted_output.get("machine_learning_RM_extracted_infos").get("all_tool_calls")
                time_conclusion = extracted_output.get("machine_learning_RM_extracted_infos").get("time_conclusion")
                accuracy_conclusion = extracted_output.get("machine_learning_RM_extracted_infos").get("accuracy_conclusion")

                score = 0.0
                # format score
                if all_tool_responses != None : 
                    score += 0.05
                if time_conclusion != None : 
                    score += 0.025
                if accuracy_conclusion != None : 
                    score += 0.025
                
                if all_tool_responses == None:
                    print("[Debug LLMTimer Reward Calculator] all tool calls is None, return score.")
                    return score

                for response in all_tool_responses:
                    match = re.search(ACCURACY_PATTERN, response)
                    if match:
                        accuracy_str = match.group(1)
                        accuracy_float = float(accuracy_str)
                        extracted_accuracies.append(accuracy_float)   
                
                required_time = float(identity.get("required_time"))

                if time_conclusion:
                    time_conclusion_pattern = r"total duration:\s*(\d+(?:\.\d+)?)\s*seconds"
                    time_conclusion_match = re.search(time_conclusion_pattern, time_conclusion, re.IGNORECASE) # re.IGNORECASE 让它不区分大小写
                
                    if time_conclusion_match:
                        number_str = time_conclusion_match.group(1)
                        total_duration = float(number_str)
                    else:
                        total_duration = None  

                    if total_duration and total_duration <= (1.0 + 1e-3) * required_time :
                        score += 0.1
                    else :
                        print("[Debug LLMTimer Reward Calculator] total duration is greater than required time.")
                
                if len(extracted_accuracies) == 0:
                    print("[Debug LLMTimer Reward Calculator] no accuracy found, return score.")
                    return score
                else:
                    score += 0.7 * max(extracted_accuracies)
                
                max_exploration_score = 0.05
                exploration_score = 0.0
                if len(extracted_accuracies) > 1: # 鼓励step之间有提升的尝试
                    for i in range(len(extracted_accuracies) - 1):
                        if extracted_accuracies[i] < extracted_accuracies[i + 1]:
                            exploration_score += 0.01
                            exploration_score = min(exploration_score, max_exploration_score)
                    if max(extracted_accuracies) > mean(extracted_accuracies): # 如果最高分大于平均分，则给予额外的奖励
                        exploration_score += 0.05

                score += exploration_score

                
                return score

    @classmethod
    def _calculate_score(cls, extracted_output: float, identity: dict, **kwargs) -> float:
        
        pass


if __name__ == "__main__":
    output_str = "<think>I need to solve this step by step. First, I'll analyze the function $ f(x) = |x+2| - |x-3| - a $. It involves absolute values, so I need to consider different cases based on the critical points $ x = -2 $ and $ x = 3 $, where the expressions inside the absolute values change sign.\n\nThe function changes its form in three intervals: $ x < -2 $, $ -2 \\leq x < 3 $, and $ x \\geq 3 $. I'll write piecewise definitions for each interval.\n\nFor $ x < -2 $:\n- $ |x+2| = -(x+2) = -x - 2 $\n- $ |x-3| = -(x-3) = -x + 3 $\nSo, $ f(x) = (-x - 2) - (-x + 3) - a = -x - 2 + x - 3 - a = -5 - a $\n\nFor $ -2 \\leq x < 3 $:\n- $ |x+2| = x+2 $\n- $ |x-3| = -(x-3) = -x + 3 $\nSo, $ f(x) = (x+2) - (-x + 3) - a = x + 2 + x - 3 - a = 2x - 1 - a $\n\nFor $ x \\geq 3 $:\n- $ |x+2| = x+2 $\n- $ |x-3| = x-3 $\nSo, $ f(x) = (x+2) - (x-3) - a = x+2 - x + 3 - a = 5 - a $\n\nNow, I have:\n- $ f(x) = -5 - a $ for $ x < -2 $\n- $ f(x) = 2x - 1 - a $ for $ -2 \\leq x < 3 $\n- $ f(x) = 5 - a $ for $ x \\geq 3 $\n\nI need to find the maximum value when $ a = 1 $, and then find the range of $ a $ such that $ f(x) \\leq \\frac{4}{a} $ for all $ x \\in \\mathbb{R} $.\n\nFirst, for $ a = 1 $:\n- $ f(x) = -5 - 1 = -6 $ for $ x < -2 $\n- $ f(x) = 2x - 1 - 1 = 2x - 2 $ for $ -2 \\leq x < 3 $\n- $ f(x) = 5 - 1 = 4 $ for $ x \\geq 3 $\n\nThe function $ 2x - 2 $ is increasing in $ [-2, 3) $. At $ x = -2 $, $ f(-2) = 2(-2) - 2 = -4 - 2 = -6 $. As $ x \\to 3^- $, $ f(x) \\to 2(3) - 2 = 6 - 2 = 4 $. So the maximum value in this interval is 4 (approached as $ x \\to 3^- $), but at $ x = 3 $, $ f(3) = 4 $ from the third piece. So the maximum value is 4.\n\nFor part (II), $ f(x) \\leq \\frac{4}{a} $ for all $ x $. From the piecewise form, the maximum values of $ f(x) $ in each interval are:\n- For $ x < -2 $, $ f(x) = -5 - a $, which is constant and clearly $ \\leq 5 - a $\n- For $ -2 \\leq x < 3 $, $ f(x) = 2x - 1 - a $, which is linear with maximum at $ x = 3^- $, giving $ 4 - a $\n- For $ x \\geq 3 $, $ f(x) = 5 - a $, constant\n\nSo the maximum value of $ f(x) $ is $ 5 - a $, which occurs as $ x \\to \\pm \\infty $? Wait, no. In the third interval $ x \\geq 3 $, $ f(x) = 5 - a $, constant. In the first interval $ x < -2 $, $ f(x) = -5 - a $. So the maximum possible value is $ 5 - a $, since $ 5 - a > -5 - a $. And in the middle interval, it approaches $ 4 - a $ as $ x \\to 3^- $, but $ 4 - a < 5 - a $, so indeed $ 5 - a $ is the global maximum.\n\nTherefore, $ f(x) \\leq \\frac{4}{a} $ for all $ x $ if and only if $ 5 - a \\leq \\frac{4}{a} $.\n\nSo, $ 5 - a \\leq \\frac{4}{a} $\n\nMultiply both sides by $ a $, but I need to be careful about the sign of $ a $. Since $ \\frac{4}{a} $ appears, $ a \\neq 0 $, and for the inequality to make sense, likely $ a > 0 $, because if $ a < 0 $, $ \\frac{4}{a} $ is negative, and $ 5 - a > 5 $, so it can't be $ \\leq \\frac{4}{a} $. Let's check: if $ a < 0 $, left side $ 5 - a > 5 > 0 $, right side $ \\frac{4}{a} < 0 $, so $ > $, not $ \\leq $. So $ a > 0 $ is necessary.\n\nNow, $ 5 - a \\leq \\frac{4}{a} $\n\nMultiply both sides by $ a $: $ (5 - a)a \\leq 4 $\n\n$ 5a - a^2 \\leq 4 $\n\n$ -a^2 + 5a - 4 \\leq 0 $\n\nMultiply by -1 (reverse inequality): $ a^2 - 5a + 4 \\geq 0 $\n\n$ (a - 1)(a - 4) \\geq 0 $\n\nSo $ a \\leq 1 $ or $ a \\geq 4 $\n\nBut earlier I assumed $ a > 0 $. So $ a \\in (0, 1] \\cup [4, \\infty) $\n\nBut wait, when $ a = 1 $, $ \\frac{4}{a} = 4 $, and maximum $ f(x) = 5 - 1 = 4 $, so equality holds. When $ a = 4 $, $ \\frac{4}{a} = 1 $, and maximum $ f(x) = 5 - 4 = 1 $, so equality again. So it should be okay.\n\nBut is $ a = 0 $ allowed? $ a = 0 $, then $ f(x) = |x+2| - |x-3| $, and $ \\frac{4}{a} $ undefined, so no.\n\nAlso, what about $ a < 0 $? We saw it's impossible because left side $ > 5 > 0 $, right side $ < 0 $.\n\nBut let's verify with a small $ a $. Suppose $ a = 0.5 $, then $ \\frac{4}{a} = 8 $, maximum $ f(x) = 5 - 0.5 = 4.5 \\leq 8 $, yes.\n\n$ a = 0.1 $, $ \\frac{4}{0.1} = 40 $, $ 5 - 0.1 = 4.9 \\leq 40 $, yes.\n\n$ a = 1.5 $, $ \\frac{4}{1.5} \\approx 2.666 $, $ 5 - 1.5 = 3.5 > 2.666 $, so not $ \\leq $, so $ a = 1.5 $ not in the range. $ a = 4.5 $, $ \\frac{4}{4.5} \\approx 0.888 $, $ 5 - 4.5 = 0.5 \\leq 0.888 $, yes.\n\nSo the range is $ a > 0 $ and $ (a \\leq 1 $ or $ a \\geq 4) $, so $ (0, 1] \\cup [4, \\infty) $\n\nBut let's double-check if $ a = 0 $ is excluded, yes.\n\nAlso, what about $ a = 1 $ and $ a = 4 $, they work, as we saw.\n\nSo final answers:\n(I) When $ a = 1 $, maximum value is $ 4 $\n(II) Range of $ a $: $ (0, 1] \\cup [4, \\infty) $\n\nNow, I should express this neatly.\n\n<summary>\nFor part (I), when $ a = 1 $, the maximum value of $ f(x) $ is 4.  \nFor part (II), the range of $ a $ is $ (0, 1] \\cup [4, \\infty) $.  \n</summary>\n\n<conclusion>total duration: 3.08 seconds</conclusion>\n\n<answer>$\\boxed{4}$</answer>\n<answer>$\\boxed{N = m(a + g)}$</answer>"
    
    extracted_output = LlmTimerRewardCalculator.extract_output(output_str)
    print(extracted_output)

    train_data_path = "./data/train.jsonl"

    # print("loading train data...")
    # def load_from_jsonl(local_load_path, num_samples=None):
    #     data_list = []
    #     count = 0
    #     with open(local_load_path, 'r') as f:
    #         for line in f:
    #             data_list.append(json.loads(line))
    #             count += 1
    #             if num_samples is not None and count >= num_samples:
    #                 break
    #     return data_list
    
    # train_data = load_from_jsonl(train_data_path, 10)
    # first_item = train_data[0]
    # print(first_item)
    # identity = first_item["reward_model"]['ground_truth']
    # identity = {
    #     "data_type": "jericho",
    #     "game_name": "dragon",
    #     "max_score": 25,
    #     "required_time": 357,
    #     "timer_mode": "static",
    #     "timer_speed_factor": 0.062263436157025065,
    #     "with_time_limit": False
    # }
# $\boxed{N = m(a + g)}$, identity answer: N = mg + ma
    identity = {
        "answer": "N = mg + ma",
        "data_type": "general_reasoning",
        "game_name": None,
        "max_score": None,
        "required_time": 10.72,
        "single_round": None,
        "task_name": None,
        "timer_mode": "static",
        "timer_speed_factor": 1.49,
        "with_time_limit": True
    }

    print(LlmTimerRewardCalculator._verify_correction(extracted_output, identity))

    # python -m internbootcamp.bootcamps.Basic_LLM_timer.Basic_timer_reward_calculator