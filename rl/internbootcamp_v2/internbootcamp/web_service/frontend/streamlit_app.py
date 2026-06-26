"""
InternBootcamp 评测平台 - Streamlit 前端
"""
import streamlit as st
import time
import json
from datetime import datetime
from pathlib import Path
from internbootcamp.web_service.frontend.utils.api_client import APIClient
from internbootcamp.web_service.frontend.utils.visualizations import (
    create_score_distribution_chart,
    create_token_usage_chart,
    create_turns_distribution_chart,
    create_error_type_pie_chart,
    create_data_source_comparison_chart,
    create_score_boxplot,
)

# 页面配置
st.set_page_config(
    page_title="InternBootcamp 评测平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 API 客户端
@st.cache_resource
def get_api_client():
    """获取 API 客户端（缓存）"""
    backend_url = st.secrets.get("BACKEND_URL", "http://localhost:8000")
    return APIClient(base_url=backend_url)

api_client = get_api_client()

# 初始化 session state
if "current_task_id" not in st.session_state:
    st.session_state.current_task_id = None
if "sample_offset" not in st.session_state:
    st.session_state.sample_offset = 0
if "all_samples" not in st.session_state:
    st.session_state.all_samples = []
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# 侧边栏 - 导航
st.sidebar.title("🚀 InternBootcamp")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航",
    ["📊 模型评测", "📝 数据生成", "📋 任务管理", "📁 结果分析"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 关于")
st.sidebar.info(
    "InternBootcamp 评测平台提供数据生成、模型评测与可视化分析功能。"
)

# 添加清除缓存按钮
st.sidebar.markdown("---")
if st.sidebar.button("🔄 清除缓存", help="如果遇到API错误，点击此按钮清除缓存"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.sidebar.success("✅ 缓存已清除，请刷新页面")

# ==================== 页面 1: 模型评测 ====================
if page == "📊 模型评测":
    st.title("📊 模型评测")
    
    # 创建两列布局
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("创建评测任务")
        
        with st.form("evaluation_form"):
            dataset_path = st.text_input(
                "数据集路径 *",
                placeholder="path/to/dataset.jsonl",
                help="支持 .jsonl, .json, .parquet 格式"
            )
            
            api_key = st.text_input(
                "API Key *",
                type="password",
                help="模型 API 密钥"
            )
            
            api_url = st.text_input(
                "API URL",
                placeholder="https://api.openai.com/v1",
                help="留空使用默认 OpenAI API"
            )
            
            api_model = st.text_input(
                "模型名称 *",
                value="gpt-3.5-turbo",
                help="例如: gpt-3.5-turbo, gpt-4"
            )
            
            reward_calculator_class = st.text_input(
                "奖励计算器类 *",
                placeholder="internbootcamp.bootcamps.example_bootcamp.example_reward_calculator.ExampleRewardCalculator",
                help="奖励计算器的完整类路径"
            )
            
            with st.expander("高级配置", expanded=False):
                tool_config = st.text_input(
                    "工具配置文件",
                    placeholder="path/to/tool_config.yaml"
                )
                
                interaction_config = st.text_input(
                    "交互配置文件",
                    placeholder="path/to/interaction_config.yaml"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    max_assistant_turns = st.number_input(
                        "最大 Assistant 轮次",
                        min_value=1,
                        max_value=100,
                        value=5
                    )
                with col_b:
                    max_user_turns = st.number_input(
                        "最大 User 轮次",
                        min_value=1,
                        max_value=100,
                        value=20
                    )
                
                max_concurrent = st.number_input(
                    "最大并发数",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="同时处理的样本数量"
                )
                
                api_extra_params = st.text_area(
                    "额外模型参数（JSON）",
                    placeholder='{"temperature": 0.7, "max_tokens": 2048}',
                    help="JSON 格式的额外参数"
                )
                
                api_extra_headers = st.text_input(
                    "额外请求头",
                    placeholder="Authorization:Bearer token,Custom-Header:Value",
                    help="格式: key1:value1,key2:value2"
                )
                
                tokenizer_path = st.text_input(
                    "Tokenizer 路径",
                    placeholder="path/to/tokenizer",
                    help="用于计算 Token 数的 tokenizer 路径（可选）"
                )
                
                verify_correction_kwargs = st.text_area(
                    "验证校正参数（JSON）",
                    placeholder='{"key": "value"}',
                    help="验证校正的额外参数，JSON 格式（可选）"
                )
            
            submitted = st.form_submit_button("🚀 开始评测", width='stretch')
            
            if submitted:
                if not all([dataset_path, api_key, api_model, reward_calculator_class]):
                    st.error("请填写所有必填字段！")
                else:
                    config = {
                        "dataset_path": dataset_path,
                        "api_key": api_key,
                        "api_url": api_url or None,
                        "api_model": api_model,
                        "reward_calculator_class": reward_calculator_class,
                        "tool_config": tool_config or None,
                        "interaction_config": interaction_config or None,
                        "max_assistant_turns": max_assistant_turns,
                        "max_user_turns": max_user_turns,
                        "max_concurrent": max_concurrent,
                        "api_extra_params": api_extra_params or None,
                        "api_extra_headers": api_extra_headers or None,
                        "tokenizer_path": tokenizer_path or None,
                        "verify_correction_kwargs": verify_correction_kwargs or None,
                    }
                    
                    with st.spinner("正在创建评测任务..."):
                        result = api_client.create_evaluation_task(config)
                        
                        if "error" in result:
                            st.error(f"创建任务失败: {result['error']}")
                        else:
                            st.success(f"任务创建成功！任务 ID: {result['task_id']}")
                            st.session_state.current_task_id = result['task_id']
                            st.session_state.sample_offset = 0
                            st.session_state.all_samples = []
                            st.rerun()
    
    with col2:
        st.subheader("任务监控与结果")
        
        # 任务选择器
        if st.session_state.current_task_id:
            selected_task_id = st.session_state.current_task_id
        else:
            tasks_response = api_client.list_tasks(task_type="evaluation")
            tasks = tasks_response.get("tasks", [])
            
            if tasks:
                task_options = {
                    f"{t['task_id'][:8]}... ({t['status']}, {t['created_at'][:19]})": t['task_id']
                    for t in tasks
                }
                selected_label = st.selectbox("选择任务", list(task_options.keys()))
                selected_task_id = task_options[selected_label]
                st.session_state.current_task_id = selected_task_id
            else:
                st.info("暂无评测任务，请先创建任务。")
                selected_task_id = None
        
        if selected_task_id:
            # 获取任务详情
            task = api_client.get_task(selected_task_id)
            
            if "error" not in task:
                # 显示任务状态
                status_colors = {
                    "pending": "🟡",
                    "running": "🟢",
                    "completed": "✅",
                    "failed": "❌",
                    "cancelled": "⚪"
                }
                
                col_status, col_progress, col_samples = st.columns(3)
                with col_status:
                    st.metric("状态", f"{status_colors.get(task['status'], '⚪')} {task['status']}")
                with col_progress:
                    st.metric("进度", f"{task.get('progress', 0):.1f}%")
                with col_samples:
                    st.metric("样本数", f"{task.get('completed_samples', 0)}/{task.get('total_samples', 0)}")
                
                # 自动刷新控制
                col_refresh1, col_refresh2 = st.columns([1, 3])
                with col_refresh1:
                    auto_refresh = st.checkbox("自动刷新", value=st.session_state.auto_refresh)
                    st.session_state.auto_refresh = auto_refresh
                with col_refresh2:
                    if st.button("🔄 手动刷新"):
                        st.rerun()
                
                # 如果任务正在运行，启用自动刷新
                if task['status'] == "running" and auto_refresh:
                    time.sleep(2)
                    st.rerun()
                
                # 标签页
                tab1, tab2, tab3, tab4 = st.tabs(["📊 汇总统计", "📝 实时结果", "📈 可视化分析", "📋 日志"])
                
                with tab1:
                    st.subheader("汇总统计")
                    
                    if task['status'] in ["completed", "running"]:
                        summary_response = api_client.get_evaluation_summary(selected_task_id)
                        
                        if "error" not in summary_response:
                            summary = summary_response.get("summary", {})
                            
                            # 总体指标卡片
                            st.markdown("#### 总体指标")
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            with metric_col1:
                                st.metric("总样本数", summary.get("total_samples", 0))
                            with metric_col2:
                                st.metric("成功率", f"{summary.get('success_rate', 0):.1%}")
                            with metric_col3:
                                st.metric("平均分", f"{summary.get('avg_score', 0):.4f}")
                            with metric_col4:
                                st.metric("错误数", summary.get("error_count", 0))
                            
                            # Token 使用统计
                            st.markdown("#### Token 使用统计")
                            token_col1, token_col2, token_col3 = st.columns(3)
                            with token_col1:
                                st.metric("Prompt Tokens", f"{summary.get('avg_prompt_tokens', 0):.0f}")
                            with token_col2:
                                st.metric("Completion Tokens", f"{summary.get('avg_completion_tokens', 0):.0f}")
                            with token_col3:
                                st.metric("Total Tokens", f"{summary.get('avg_tokens', 0):.0f}")
                            
                            # 数据源统计
                            st.markdown("#### 数据源统计")
                            data_sources = summary_response.get("data_sources", [])
                            if data_sources:
                                df_data_sources = []
                                for ds in data_sources:
                                    df_data_sources.append({
                                        "数据源": ds["data_source"],
                                        "样本数": ds["total_count"],
                                        "成功率": f"{ds['success_rate']:.1%}",
                                        "平均分": f"{ds['avg_score']:.4f}",
                                        "最高分": f"{ds['max_score']:.4f}",
                                        "最低分": f"{ds['min_score']:.4f}",
                                        "平均轮次": f"{ds['avg_assistant_turns']:.2f}",
                                        "平均工具调用": f"{ds['avg_tool_calls']:.2f}",
                                    })
                                st.dataframe(df_data_sources, width='stretch')
                            
                            # 生成器统计
                            st.markdown("#### 生成器统计")
                            generators = summary_response.get("generators", {})
                            if generators:
                                for data_source, gen_list in generators.items():
                                    with st.expander(f"📁 {data_source}", expanded=False):
                                        df_generators = []
                                        for gen in gen_list:
                                            df_generators.append({
                                                "生成器": gen["generator_name"],
                                                "样本数": gen["total_count"],
                                                "成功率": f"{gen['success_rate']:.1%}",
                                                "平均分": f"{gen['avg_score']:.4f}",
                                                "最高分": f"{gen['max_score']:.4f}",
                                                "最低分": f"{gen['min_score']:.4f}",
                                            })
                                        st.dataframe(df_generators, width='stretch')
                            
                            # 错误分析
                            errors = summary_response.get("errors", [])
                            if errors:
                                st.markdown("#### 错误分析")
                                with st.expander(f"⚠️ 查看错误详情 ({len(errors)} 个错误)", expanded=False):
                                    for i, error in enumerate(errors[:20]):  # 限制显示前20个
                                        st.text(f"{i+1}. [{error['data_source']}] {error['error'][:100]}...")
                        else:
                            st.warning("暂无汇总数据")
                    else:
                        st.info("任务未完成，暂无汇总数据")
                
                with tab2:
                    st.subheader("实时结果流")
                    
                    if task['status'] in ["completed", "running"]:
                        # 加载增量结果
                        results_response = api_client.get_evaluation_results(
                            selected_task_id,
                            offset=st.session_state.sample_offset,
                            limit=10
                        )
                        
                        if "error" not in results_response:
                            new_samples = results_response.get("samples", [])
                            
                            if new_samples:
                                # 追加新样本到全局列表
                                st.session_state.all_samples.extend(new_samples)
                                st.session_state.sample_offset = results_response.get("offset", st.session_state.sample_offset)
                            
                            # 显示所有样本（倒序，最新的在前）
                            if st.session_state.all_samples:
                                st.markdown(f"**已加载 {len(st.session_state.all_samples)} 个样本**")
                                
                                for i, sample in enumerate(reversed(st.session_state.all_samples)):
                                    with st.expander(
                                        f"{'✅' if sample.get('success') else '❌'} Sample #{sample.get('sample_id', 'N/A')} | "
                                        f"Score: {sample.get('score', 0):.4f} | "
                                        f"Data Source: {sample.get('data_source', 'N/A')}",
                                        expanded=False
                                    ):
                                        # 检测格式类型
                                        format_type = sample.get('format_type', 'standard')
                                        
                                        # 基本信息卡片
                                        st.markdown("### 📋 基本信息")
                                        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                                        with col_info1:
                                            st.metric("样本ID", sample.get('sample_id', 'N/A'))
                                        with col_info2:
                                            st.metric("得分", f"{sample.get('score', 0):.4f}")
                                        with col_info3:
                                            status_icon = "✅ 成功" if sample.get('success') else "❌ 失败"
                                            st.metric("状态", status_icon)
                                        with col_info4:
                                            st.metric("数据源", sample.get('data_source', 'N/A'))
                                        
                                        # 生成器信息
                                        if sample.get('generator_name'):
                                            st.markdown(f"**生成器**: `{sample.get('generator_name')}`")
                                        
                                        # 轮次统计
                                        st.markdown("### 🔄 轮次统计")
                                        col_turn1, col_turn2, col_turn3 = st.columns(3)
                                        with col_turn1:
                                            st.metric("Assistant 轮次", sample.get('assistant_turns', 0))
                                        with col_turn2:
                                            st.metric("工具调用次数", sample.get('tool_calls', 0))
                                        with col_turn3:
                                            st.metric("交互轮次", sample.get('interaction_turns', 0))
                                        
                                        # Token 使用详情
                                        st.markdown("### 💰 Token 使用详情")
                                        col_token1, col_token2, col_token3 = st.columns(3)
                                        with col_token1:
                                            st.metric("Prompt Tokens", sample.get('prompt_tokens', 0))
                                        with col_token2:
                                            st.metric("Completion Tokens", sample.get('completion_tokens', 0))
                                        with col_token3:
                                            st.metric("Total Tokens", sample.get('total_tokens', 0))
                                        
                                        # 输出对比
                                        st.markdown("### 📊 输出对比")
                                        col_output1, col_output2 = st.columns(2)
                                        
                                        with col_output1:
                                            st.markdown("**🤖 模型输出（提取后）:**")
                                            if sample.get('extracted_output') is not None:
                                                st.json(sample['extracted_output'])
                                            else:
                                                st.info("无输出")
                                        
                                        with col_output2:
                                            st.markdown("**✅ 标准答案:**")
                                            if sample.get('ground_truth'):
                                                st.json(sample['ground_truth'])
                                            else:
                                                st.info("无标准答案")
                                        
                                        # 错误信息（如果失败）
                                        if not sample.get('success'):
                                            st.markdown("### ⚠️ 错误信息")
                                            if sample.get('error'):
                                                st.error(sample['error'])
                                            else:
                                                st.warning("样本失败但未提供错误信息")
                                        
                                        # 根据格式类型显示不同的详细信息
                                        if format_type == 'iteration':
                                            # 新格式：显示迭代信息
                                            st.markdown("### 🔄 迭代详情")
                                            iterations = sample.get('iterations', [])
                                            iteration_scores = sample.get('iteration_scores', [])
                                            best_iteration_idx = sample.get('best_iteration_idx', -1)
                                            best_score = sample.get('best_score', sample.get('score', 0))
                                            final_score = sample.get('final_score', 0)
                                            
                                            if iterations:
                                                # 显示关键信息
                                                col_iter1, col_iter2, col_iter3 = st.columns(3)
                                                with col_iter1:
                                                    st.metric("总迭代次数", len(iterations))
                                                with col_iter2:
                                                    st.metric("最高分", f"{best_score:.4f}", 
                                                             help=f"出现在第 {best_iteration_idx + 1} 次迭代" if best_iteration_idx >= 0 else None)
                                                with col_iter3:
                                                    delta = final_score - best_score if final_score != best_score else None
                                                    st.metric("最终分数", f"{final_score:.4f}", 
                                                             delta=f"{delta:.4f}" if delta else None,
                                                             help="最后一次迭代的分数")
                                                
                                                # 显示每次迭代的得分
                                                if iteration_scores:
                                                    st.markdown("**各迭代得分**:")
                                                    score_cols = st.columns(min(len(iteration_scores), 6))
                                                    for idx, score in enumerate(iteration_scores):
                                                        with score_cols[idx % 6]:
                                                            # 标记最高分
                                                            is_best = (idx == best_iteration_idx)
                                                            label = f"{'🏆 ' if is_best else ''}迭代 {idx + 1}"
                                                            st.metric(label, f"{score:.4f}")
                                                
                                                # 显示每次迭代的详细信息
                                                for iter_idx, iteration in enumerate(iterations):
                                                    is_best = (iter_idx == best_iteration_idx)
                                                    iter_score = iteration_scores[iter_idx] if iter_idx < len(iteration_scores) else 0
                                                    title = f"{'🏆 ' if is_best else ''}迭代 {iter_idx + 1} 详情 (得分: {iter_score:.4f})"
                                                    # 默认展开最佳迭代
                                                    with st.expander(title, expanded=is_best):
                                                        # 显示该迭代的对话历史
                                                        messages = iteration.get('messages', [])
                                                        if messages:
                                                            st.markdown(f"**对话历史 ({len(messages)} 条消息)**:")
                                                            for msg_idx, msg in enumerate(messages):
                                                                role = msg.get('role', 'unknown')
                                                                content = msg.get('content', '')
                                                                
                                                                # 根据角色设置样式
                                                                if role == 'user':
                                                                    st.markdown(f"**👤 User (消息 {msg_idx + 1}):**")
                                                                elif role == 'assistant':
                                                                    st.markdown(f"**🤖 Assistant (消息 {msg_idx + 1}):**")
                                                                elif role == 'tool':
                                                                    st.markdown(f"**🔧 Tool (消息 {msg_idx + 1}):**")
                                                                else:
                                                                    st.markdown(f"**📝 {role.capitalize()} (消息 {msg_idx + 1}):**")
                                                                
                                                                # 显示内容
                                                                if content:
                                                                    st.text_area(
                                                                        f"iter_{iter_idx}_msg_{msg_idx}",
                                                                        content,
                                                                        height=150,
                                                                        key=f"realtime_iter_{i}_{iter_idx}_{msg_idx}",
                                                                        label_visibility="collapsed"
                                                                    )
                                                                
                                                                # 显示 tool_calls（如果有）
                                                                if msg.get('tool_calls'):
                                                                    st.markdown("**🔧 工具调用:**")
                                                                    for tool_call in msg['tool_calls']:
                                                                        if isinstance(tool_call, dict):
                                                                            tool_name = tool_call.get('function', {}).get('name', 'unknown')
                                                                            tool_args = tool_call.get('function', {}).get('arguments', '{}')
                                                                            st.code(f"Function: {tool_name}\nArguments: {tool_args}", language="json")
                                                                
                                                                st.markdown("---")
                                                        
                                                        # 显示该迭代的统计信息
                                                        turn_record = iteration.get('turn_record', {})
                                                        if turn_record:
                                                            st.markdown("**轮次统计**:")
                                                            for turn_key, turn_data in turn_record.items():
                                                                st.write(f"- {turn_key}: {turn_data}")
                                            else:
                                                st.info("无迭代信息")
                                        else:
                                            # 标准格式：显示完整对话历史
                                            st.markdown("### 💬 完整对话历史")
                                            if sample.get('messages'):
                                                with st.expander(f"查看对话历史 ({len(sample['messages'])} 条消息)", expanded=False):
                                                    for idx, msg in enumerate(sample['messages']):
                                                        role = msg.get('role', 'unknown')
                                                        content = msg.get('content', '')
                                                        
                                                        # 根据角色设置样式
                                                        if role == 'user':
                                                            st.markdown(f"**👤 User (消息 {idx + 1}):**")
                                                        elif role == 'assistant':
                                                            st.markdown(f"**🤖 Assistant (消息 {idx + 1}):**")
                                                        elif role == 'system':
                                                            st.markdown(f"**⚙️ System (消息 {idx + 1}):**")
                                                        elif role == 'tool':
                                                            st.markdown(f"**🔧 Tool (消息 {idx + 1}):**")
                                                        else:
                                                            st.markdown(f"**📝 {role.capitalize()} (消息 {idx + 1}):**")
                                                        
                                                        # 显示内容
                                                        if content:
                                                            st.text_area(
                                                                f"content_{idx}",
                                                                content,
                                                                height=100,
                                                                key=f"sample_{sample.get('sample_id')}_{idx}",
                                                                label_visibility="collapsed"
                                                            )
                                                        
                                                        # 显示 tool_calls（如果有）
                                                        if msg.get('tool_calls'):
                                                            st.markdown("**🔧 工具调用:**")
                                                            for tool_call in msg['tool_calls']:
                                                                tool_name = tool_call.get('function', {}).get('name', 'unknown')
                                                                tool_args = tool_call.get('function', {}).get('arguments', '{}')
                                                                st.code(f"Function: {tool_name}\nArguments: {tool_args}", language="json")
                                                        
                                                        st.markdown("---")
                                            else:
                                                st.info("无对话历史")
                                        
                                        # 原始数据（供调试使用）
                                        with st.expander("🔍 查看原始 JSON 数据", expanded=False):
                                            st.json(sample)
                            else:
                                st.info("暂无结果数据")
                            
                            # 如果还有更多数据，显示加载按钮
                            if results_response.get("has_more", False):
                                if st.button("📥 加载更多"):
                                    st.rerun()
                        else:
                            st.warning("无法加载结果数据")
                    else:
                        st.info("任务未开始或已完成")
                
                with tab3:
                    st.subheader("可视化分析")
                    
                    if task['status'] in ["completed", "running"] and st.session_state.all_samples:
                        # 分数分布图
                        st.markdown("#### 分数分布")
                        score_fig = create_score_distribution_chart(st.session_state.all_samples)
                        if score_fig:
                            st.plotly_chart(score_fig, width='stretch')
                        
                        # Token 使用图
                        st.markdown("#### Token 使用统计")
                        token_fig = create_token_usage_chart(st.session_state.all_samples)
                        if token_fig:
                            st.plotly_chart(token_fig, width='stretch')
                        
                        # 轮次分布图
                        st.markdown("#### 轮次与工具调用分布")
                        turns_fig = create_turns_distribution_chart(st.session_state.all_samples)
                        if turns_fig:
                            st.plotly_chart(turns_fig, width='stretch')
                        
                        # 数据源对比（如果有汇总数据）
                        summary_response = api_client.get_evaluation_summary(selected_task_id)
                        if "error" not in summary_response:
                            data_sources = summary_response.get("data_sources", [])
                            if data_sources:
                                st.markdown("#### 数据源对比")
                                ds_fig = create_data_source_comparison_chart(data_sources)
                                if ds_fig:
                                    st.plotly_chart(ds_fig, width='stretch')
                            
                            # 错误类型分布
                            errors = summary_response.get("errors", [])
                            if errors:
                                st.markdown("#### 错误类型分布")
                                error_fig = create_error_type_pie_chart(errors)
                                if error_fig:
                                    st.plotly_chart(error_fig, width='stretch')
                    else:
                        st.info("暂无数据可视化")
                
                with tab4:
                    st.subheader("任务日志")
                    
                    log_lines = st.slider("显示行数", 10, 500, 100)
                    
                    logs_response = api_client.get_task_logs(selected_task_id, tail=log_lines)
                    logs = logs_response.get("logs", "")
                    
                    if logs:
                        st.text_area("日志输出", logs, height=400)
                    else:
                        st.info("暂无日志")
                
                # 操作按钮
                st.markdown("---")
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                
                with col_btn1:
                    if task['status'] == "running":
                        if st.button("⏸️ 取消任务", width='stretch'):
                            api_client.cancel_task(selected_task_id)
                            st.success("任务已取消")
                            st.rerun()
                
                with col_btn2:
                    if task['status'] == "completed":
                        if st.button("📥 下载 JSONL", width='stretch'):
                            data = api_client.download_evaluation_results(selected_task_id, "jsonl")
                            if data:
                                st.download_button(
                                    "💾 保存文件",
                                    data=data,
                                    file_name=f"results_{selected_task_id[:8]}.jsonl",
                                    mime="application/json"
                                )
                
                with col_btn3:
                    if task['status'] == "completed":
                        if st.button("📥 下载 CSV", width='stretch'):
                            data = api_client.download_evaluation_results(selected_task_id, "csv")
                            if data:
                                st.download_button(
                                    "💾 保存文件",
                                    data=data,
                                    file_name=f"results_{selected_task_id[:8]}.csv",
                                    mime="text/csv"
                                )
                
                with col_btn4:
                    if st.button("🗑️ 删除任务", width='stretch'):
                        api_client.delete_task(selected_task_id)
                        st.session_state.current_task_id = None
                        st.session_state.sample_offset = 0
                        st.session_state.all_samples = []
                        st.success("任务已删除")
                        st.rerun()

# ==================== 页面 2: 数据生成 ====================
elif page == "📝 数据生成":
    st.title("📝 数据生成")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("创建数据生成任务")
        
        with st.form("data_generation_form"):
            instruction_config = st.text_input(
                "指令配置文件 *",
                placeholder="path/to/instruction_config.yaml",
                help="指令生成器配置文件路径"
            )
            
            output_dir = st.text_input(
                "输出目录",
                value="outputs/data_generation",
                help="生成数据的输出目录"
            )
            
            split_samples = st.text_input(
                "数据集划分",
                value="train:100,test:10",
                help="格式: train:100,test:10"
            )
            
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                shuffle = st.checkbox("打乱数据", value=True)
                gen_parquet = st.checkbox("生成 Parquet", value=False)
            with col_opt2:
                no_tool = st.checkbox("不使用工具", value=False)
                no_interaction = st.checkbox("不使用交互", value=False)
            
            submitted = st.form_submit_button("🚀 开始生成", width='stretch')
            
            if submitted:
                if not instruction_config:
                    st.error("请填写指令配置文件路径！")
                else:
                    config = {
                        "instruction_config": instruction_config,
                        "output_dir": output_dir,
                        "split_samples": split_samples,
                        "shuffle": shuffle,
                        "gen_parquet": gen_parquet,
                        "no_tool": no_tool,
                        "no_interaction": no_interaction,
                    }
                    
                    with st.spinner("正在创建数据生成任务..."):
                        result = api_client.create_data_generation_task(config)
                        
                        if "error" in result:
                            st.error(f"创建任务失败: {result['error']}")
                        else:
                            st.success(f"任务创建成功！任务 ID: {result['task_id']}")
    
    with col2:
        st.subheader("数据生成任务历史")
        
        tasks_response = api_client.list_tasks(task_type="data_generation")
        tasks = tasks_response.get("tasks", [])
        
        if tasks:
            for task in tasks[:10]:  # 显示最近10个任务
                with st.expander(
                    f"{'✅' if task['status'] == 'completed' else '🔄'} {task['task_id'][:8]}... | {task['status']} | {task['created_at'][:19]}",
                    expanded=False
                ):
                    st.text(f"任务ID: {task['task_id']}")
                    st.text(f"状态: {task['status']}")
                    st.text(f"创建时间: {task['created_at']}")
                    st.text(f"结果路径: {task.get('result_path', 'N/A')}")
                    
                    if st.button(f"查看日志", key=f"log_{task['task_id']}"):
                        logs_response = api_client.get_task_logs(task['task_id'])
                        logs = logs_response.get("logs", "")
                        st.text_area("日志", logs, height=300, key=f"log_area_{task['task_id']}")
        else:
            st.info("暂无数据生成任务")

# ==================== 页面 3: 任务管理 ====================
elif page == "📋 任务管理":
    st.title("📋 任务管理")
    
    # 筛选器
    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        filter_type = st.selectbox("任务类型", ["全部", "评测", "数据生成"])
    
    task_type_map = {
        "全部": None,
        "评测": "evaluation",
        "数据生成": "data_generation"
    }
    
    # 获取任务列表
    tasks_response = api_client.list_tasks(task_type=task_type_map[filter_type])
    tasks = tasks_response.get("tasks", [])
    
    if tasks:
        st.markdown(f"**共 {len(tasks)} 个任务**")
        
        # 创建表格数据
        table_data = []
        for task in tasks:
            table_data.append({
                "任务ID": task['task_id'][:12] + "...",
                "类型": task['task_type'],
                "状态": task['status'],
                "进度": f"{task.get('progress', 0):.1f}%",
                "创建时间": task['created_at'][:19],
            })
        
        st.dataframe(table_data, width='stretch')
        
        # 任务详情
        st.markdown("---")
        st.subheader("任务详情")
        
        selected_task_idx = st.selectbox(
            "选择任务查看详情",
            range(len(tasks)),
            format_func=lambda i: f"{tasks[i]['task_id'][:12]}... ({tasks[i]['status']})"
        )
        
        selected_task = tasks[selected_task_idx]
        
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            st.json({
                "task_id": selected_task['task_id'],
                "task_type": selected_task['task_type'],
                "status": selected_task['status'],
                "created_at": selected_task['created_at'],
                "updated_at": selected_task['updated_at'],
            })
        with col_detail2:
            st.json({
                "progress": selected_task.get('progress', 0),
                "total_samples": selected_task.get('total_samples', 0),
                "completed_samples": selected_task.get('completed_samples', 0),
                "result_path": selected_task.get('result_path', 'N/A'),
                "log_path": selected_task.get('log_path', 'N/A'),
            })
        
        # 操作按钮
        col_op1, col_op2, col_op3 = st.columns(3)
        with col_op1:
            if selected_task['status'] == "running":
                if st.button("⏸️ 取消任务"):
                    api_client.cancel_task(selected_task['task_id'])
                    st.success("任务已取消")
                    st.rerun()
        with col_op2:
            if st.button("📋 查看日志"):
                logs_response = api_client.get_task_logs(selected_task['task_id'], tail=200)
                logs = logs_response.get("logs", "")
                st.text_area("日志输出", logs, height=400)
        with col_op3:
            if st.button("🗑️ 删除任务"):
                api_client.delete_task(selected_task['task_id'])
                st.success("任务已删除")
                st.rerun()
    else:
        st.info("暂无任务")

# ==================== 页面 4: 结果分析 ====================
elif page == "📁 结果分析":
    st.title("📁 结果分析")
    st.markdown("分析已有的评测结果文件，无需关联任务即可查看详细统计和可视化。")
    
    # 文件路径输入
    col_input1, col_input2 = st.columns([3, 1])
    
    with col_input1:
        file_path = st.text_input(
            "评测结果文件路径 (JSONL)",
            placeholder="例如: outputs/evaluation/gpt-3.5-turbo/eval_results_20240115_143022.jsonl",
            help="输入已完成评测的 JSONL 文件路径"
        )
    
    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)  # 对齐按钮
        analyze_button = st.button("🔍 开始分析", width='stretch', type="primary")
    
    # 常用文件快速选择
    with st.expander("💡 快速选择最近的评测结果", expanded=False):
        # 扫描 outputs/evaluation 目录
        eval_dir = Path("outputs/evaluation")
        if eval_dir.exists():
            jsonl_files = list(eval_dir.glob("**/*.jsonl"))
            if jsonl_files:
                # 按修改时间排序
                jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                st.markdown("**最近的评测文件：**")
                for i, file in enumerate(jsonl_files[:10]):  # 显示最近10个
                    file_info = f"📄 {file.name} ({file.parent.name})"
                    file_size = file.stat().st_size / 1024  # KB
                    file_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    
                    if st.button(
                        f"{file_info} | {file_size:.1f} KB | {file_time}",
                        key=f"file_select_{i}",
                        width='stretch'
                    ):
                        st.session_state.selected_file = str(file)
                        st.rerun()
            else:
                st.info("未找到评测结果文件")
        else:
            st.info("outputs/evaluation 目录不存在")
    
    # 使用选中的文件
    if "selected_file" in st.session_state:
        file_path = st.session_state.selected_file
        analyze_button = True
    
    # 保存当前文件路径到session_state
    if analyze_button and file_path:
        st.session_state["current_analysis_file"] = file_path
    
    # 如果有正在分析的文件，显示结果
    if "current_analysis_file" in st.session_state:
        file_path = st.session_state["current_analysis_file"]
        cache_key = f"analysis_{file_path}"
        
        # 检查文件大小（仅在首次分析时显示）
        if cache_key not in st.session_state:
            try:
                file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                if file_size > 10:
                    st.warning(f"⚠️ 文件较大 ({file_size:.1f} MB)，分析可能需要一些时间...")
            except:
                pass
        
        if cache_key not in st.session_state:
            with st.spinner(f"正在分析文件统计信息... 请稍候"):
                # 调用后端 API 分析文件（仅获取统计信息）
                response = api_client.analyze_file(file_path)
                
                if "error" in response:
                    st.error(f"分析失败: {response['error']}")
                    response = None
                else:
                    # 缓存结果
                    st.session_state[cache_key] = response
                    # 初始化样本分页状态
                    st.session_state[f"{cache_key}_sample_offset"] = 0
                    st.session_state[f"{cache_key}_samples"] = []
        else:
            response = st.session_state[cache_key]
        
        if response and "error" not in response:
            st.success("✅ 文件分析完成！")
            
            # 显示文件信息和操作按钮
            col_info1, col_info2, col_info3, col_btn1, col_btn2 = st.columns([2, 2, 2, 1, 1])
            with col_info1:
                st.metric("文件大小", f"{response.get('file_size_mb', 0):.2f} MB")
            with col_info2:
                st.metric("总样本数", response.get('total_samples', 0))
            with col_info3:
                st.metric("已分析样本", response.get('analyzed_samples', 0))
            with col_btn1:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 重新分析", help="清除缓存并重新分析文件", use_container_width=True):
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]
                    if f"{cache_key}_current_samples" in st.session_state:
                        del st.session_state[f"{cache_key}_current_samples"]
                    if "current_analysis_file" in st.session_state:
                        del st.session_state["current_analysis_file"]
                    st.rerun()
            with col_btn2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("❌ 关闭", help="返回文件选择", use_container_width=True):
                    if "current_analysis_file" in st.session_state:
                        del st.session_state["current_analysis_file"]
                    st.rerun()
            
            # 显示截断提示
            if response.get('is_truncated', False):
                st.warning(f"⚠️ 文件包含 {response.get('total_samples', 0)} 个样本，为提高性能仅分析了前 {response.get('analyzed_samples', 0)} 个样本。")
            
            st.markdown(f"**文件路径**: `{response['file_path']}`")
            
            # 创建标签页
            tab1, tab2, tab3, tab4 = st.tabs(["📊 汇总统计", "📝 样本列表", "📈 可视化分析", "⚠️ 错误分析"])
            
            with tab1:
                st.subheader("汇总统计")
                
                summary = response.get("summary", {})
                
                # 总体指标卡片
                st.markdown("#### 总体指标")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("总样本数", summary.get("total_samples", 0))
                with metric_col2:
                    st.metric("成功率", f"{summary.get('success_rate', 0):.1%}")
                with metric_col3:
                    st.metric("平均分", f"{summary.get('avg_score', 0):.4f}")
                with metric_col4:
                    st.metric("错误数", summary.get("error_count", 0))
                
                # Token 使用统计
                st.markdown("#### Token 使用统计")
                token_col1, token_col2, token_col3 = st.columns(3)
                with token_col1:
                    st.metric("平均 Prompt Tokens", f"{summary.get('avg_prompt_tokens', 0):.0f}")
                with token_col2:
                    st.metric("平均 Completion Tokens", f"{summary.get('avg_completion_tokens', 0):.0f}")
                with token_col3:
                    st.metric("平均 Total Tokens", f"{summary.get('avg_tokens', 0):.0f}")
                
                # 数据源统计
                st.markdown("#### 数据源统计")
                data_sources = response.get("data_sources", [])
                if data_sources:
                    df_data_sources = []
                    for ds in data_sources:
                        df_data_sources.append({
                            "数据源": ds["data_source"],
                            "样本数": ds["total_count"],
                            "成功率": f"{ds['success_rate']:.1%}",
                            "平均分": f"{ds['avg_score']:.4f}",
                            "最高分": f"{ds['max_score']:.4f}",
                            "最低分": f"{ds['min_score']:.4f}",
                            "平均轮次": f"{ds['avg_assistant_turns']:.2f}",
                            "平均工具调用": f"{ds['avg_tool_calls']:.2f}",
                        })
                    st.dataframe(df_data_sources, width='stretch')
                
                # 生成器统计
                st.markdown("#### 生成器统计")
                generators = response.get("generators", {})
                if generators:
                    for data_source, gen_list in generators.items():
                        with st.expander(f"📁 {data_source}", expanded=False):
                            df_generators = []
                            for gen in gen_list:
                                df_generators.append({
                                    "生成器": gen["generator_name"],
                                    "样本数": gen["total_count"],
                                    "成功率": f"{gen['success_rate']:.1%}",
                                    "平均分": f"{gen['avg_score']:.4f}",
                                    "最高分": f"{gen['max_score']:.4f}",
                                    "最低分": f"{gen['min_score']:.4f}",
                                })
                            st.dataframe(df_generators, width='stretch')
            
            with tab2:
                st.subheader("样本列表")
                
                total_samples = response.get('total_samples', 0)
                st.markdown(f"**总样本数: {total_samples}**")
                
                # 使用简单的分页器组件
                st.markdown("---")
                
                # 初始化分页状态
                page_key = f"{cache_key}_page"
                limit_key = f"{cache_key}_limit"
                sort_key = f"{cache_key}_sort"
                
                if page_key not in st.session_state:
                    st.session_state[page_key] = 1
                if limit_key not in st.session_state:
                    st.session_state[limit_key] = 20
                if sort_key not in st.session_state:
                    st.session_state[sort_key] = {"by": None, "order": "desc"}
                
                # 排序控制
                st.markdown("**排序选项**")
                col_sort1, col_sort2 = st.columns(2)
                with col_sort1:
                    sort_by = st.selectbox(
                        "排序依据",
                        options=["不排序", "按分数", "按成功状态", "按Token数"],
                        key=f"{cache_key}_sort_by_select"
                    )
                    # 映射到API参数
                    sort_by_map = {
                        "不排序": None,
                        "按分数": "score",
                        "按成功状态": "success",
                        "按Token数": "tokens"
                    }
                    sort_by_param = sort_by_map[sort_by]
                    
                with col_sort2:
                    sort_order = st.selectbox(
                        "排序顺序",
                        options=["降序", "升序"],
                        key=f"{cache_key}_sort_order_select",
                        disabled=(sort_by == "不排序")
                    )
                    sort_order_param = "desc" if sort_order == "降序" else "asc"
                
                # 检查排序是否改变
                if (sort_by_param != st.session_state[sort_key]["by"] or 
                    sort_order_param != st.session_state[sort_key]["order"]):
                    st.session_state[sort_key]["by"] = sort_by_param
                    st.session_state[sort_key]["order"] = sort_order_param
                    st.session_state[page_key] = 1  # 排序改变时重置到第一页
                    # 清除所有已加载的缓存
                    keys_to_delete = [k for k in st.session_state.keys() if k.startswith(f"{cache_key}_autoload_")]
                    for k in keys_to_delete:
                        del st.session_state[k]
                
                st.markdown("---")
                
                # 分页控制
                col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])
                
                with col_ctrl1:
                    # 每页数量选择
                    items_per_page = st.selectbox(
                        "每页显示",
                        options=[10, 20, 50, 100],
                        index=[10, 20, 50, 100].index(st.session_state[limit_key]) if st.session_state[limit_key] in [10, 20, 50, 100] else 1,
                        key=f"{cache_key}_limit_select"
                    )
                    if items_per_page != st.session_state[limit_key]:
                        st.session_state[limit_key] = items_per_page
                        st.session_state[page_key] = 1  # 重置到第一页
                
                with col_ctrl2:
                    # 页码选择器
                    total_pages = max(1, (total_samples + items_per_page - 1) // items_per_page)
                    
                    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns(5)
                    with col_nav1:
                        if st.button("⏮️", key="first_page", disabled=(st.session_state[page_key] == 1)):
                            st.session_state[page_key] = 1
                            st.rerun()
                    with col_nav2:
                        if st.button("◀️", key="prev_page", disabled=(st.session_state[page_key] == 1)):
                            st.session_state[page_key] = max(1, st.session_state[page_key] - 1)
                            st.rerun()
                    with col_nav3:
                        st.markdown(f"<div style='text-align: center; padding: 5px;'><b>{st.session_state[page_key]} / {total_pages}</b></div>", unsafe_allow_html=True)
                    with col_nav4:
                        if st.button("▶️", key="next_page", disabled=(st.session_state[page_key] >= total_pages)):
                            st.session_state[page_key] = min(total_pages, st.session_state[page_key] + 1)
                            st.rerun()
                    with col_nav5:
                        if st.button("⏭️", key="last_page", disabled=(st.session_state[page_key] >= total_pages)):
                            st.session_state[page_key] = total_pages
                            st.rerun()
                
                with col_ctrl3:
                    # 跳转到指定页
                    go_to_page = st.number_input(
                        "跳转到页",
                        min_value=1,
                        max_value=total_pages,
                        value=st.session_state[page_key],
                        key=f"{cache_key}_goto"
                    )
                    if go_to_page != st.session_state[page_key]:
                        st.session_state[page_key] = go_to_page
                        st.rerun()
                
                # 计算当前页的offset
                current_page = st.session_state[page_key]
                current_limit = st.session_state[limit_key]
                current_offset = (current_page - 1) * current_limit
                current_sort = st.session_state[sort_key]
                
                # 自动加载当前页数据（包含排序信息在缓存key中）
                auto_load_key = f"{cache_key}_autoload_{current_page}_{current_limit}_{current_sort['by']}_{current_sort['order']}"
                if auto_load_key not in st.session_state:
                    with st.spinner(f"正在加载第 {current_page} 页..."):
                        samples_response = api_client.get_file_samples(
                            file_path=file_path,
                            offset=current_offset,
                            limit=current_limit,
                            sort_by=current_sort["by"],
                            sort_order=current_sort["order"]
                        )
                        
                        if "error" not in samples_response:
                            samples = samples_response.get("samples", [])
                            st.session_state[f"{cache_key}_current_samples"] = samples
                            st.session_state[f"{cache_key}_loaded_offset"] = current_offset
                            st.session_state[f"{cache_key}_loaded_limit"] = current_limit
                            st.session_state[auto_load_key] = True
                            
                            if samples:
                                first_score = samples[0].get('score', 'N/A')
                                sort_info = f"，排序: {current_sort['by']} ({current_sort['order']})" if current_sort['by'] else ""
                                st.success(f"✅ 已加载第 {current_page} 页 ({len(samples)} 个样本，起始索引: {current_offset}，首个得分: {first_score}{sort_info})")
                        else:
                            st.error(f"加载失败: {samples_response['error']}")
                
                st.markdown("---")
                
                # 显示已加载的样本
                samples = st.session_state.get(f"{cache_key}_current_samples", [])
                loaded_offset = st.session_state.get(f"{cache_key}_loaded_offset", 0)
                loaded_limit = st.session_state.get(f"{cache_key}_loaded_limit", 0)
                
                if samples:
                    st.markdown(f"**当前显示: 第 {loaded_offset + 1} - {loaded_offset + len(samples)} 个样本** (共 {total_samples} 个)")
                        
                    # 显示样本
                    for i, sample in enumerate(samples):
                        with st.expander(
                            f"{'✅' if sample.get('success') else '❌'} Sample #{sample.get('sample_id', 'N/A')} | "
                            f"Score: {sample.get('score', 0):.4f} | "
                            f"Data Source: {sample.get('data_source', 'N/A')}",
                            expanded=False
                        ):
                            # 检测格式类型
                            format_type = sample.get('format_type', 'standard')
                                
                            # 基本信息卡片
                            st.markdown("### 📋 基本信息")
                            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                            with col_info1:
                                st.metric("样本ID", sample.get('sample_id', 'N/A'))
                            with col_info2:
                                st.metric("得分", f"{sample.get('score', 0):.4f}")
                            with col_info3:
                                status_icon = "✅ 成功" if sample.get('success') else "❌ 失败"
                                st.metric("状态", status_icon)
                            with col_info4:
                                st.metric("数据源", sample.get('data_source', 'N/A'))
                                
                            # 生成器信息
                            if sample.get('generator_name'):
                                st.markdown(f"**生成器**: `{sample.get('generator_name')}`")
                                
                            # 轮次统计
                            st.markdown("### 🔄 轮次统计")
                            col_turn1, col_turn2, col_turn3 = st.columns(3)
                            with col_turn1:
                                st.metric("Assistant 轮次", sample.get('assistant_turns', 0))
                            with col_turn2:
                                st.metric("工具调用次数", sample.get('tool_calls', 0))
                            with col_turn3:
                                st.metric("交互轮次", sample.get('interaction_turns', 0))
                                
                            # Token 使用详情
                            st.markdown("### 💰 Token 使用详情")
                            col_token1, col_token2, col_token3 = st.columns(3)
                            with col_token1:
                                st.metric("Prompt Tokens", sample.get('prompt_tokens', 0))
                            with col_token2:
                                st.metric("Completion Tokens", sample.get('completion_tokens', 0))
                            with col_token3:
                                st.metric("Total Tokens", sample.get('total_tokens', 0))
                                
                            # 输出对比
                            st.markdown("### 📊 输出对比")
                            col_output1, col_output2 = st.columns(2)
                                
                            with col_output1:
                                st.markdown("**🤖 模型输出（提取后）:**")
                                if sample.get('extracted_output') is not None:
                                    st.json(sample['extracted_output'])
                                else:
                                    st.info("无输出")
                                
                            with col_output2:
                                st.markdown("**✅ 标准答案:**")
                                if sample.get('ground_truth'):
                                    st.json(sample['ground_truth'])
                                else:
                                    st.info("无标准答案")
                                
                            # 错误信息（如果失败）
                            if not sample.get('success'):
                                st.markdown("### ⚠️ 错误信息")
                                if sample.get('error'):
                                    st.error(sample['error'])
                                else:
                                    st.warning("样本失败但未提供错误信息")
                                
                            # 根据格式类型显示不同的详细信息
                            if format_type == 'iteration':
                                # 新格式：显示迭代信息
                                st.markdown("### 🔄 迭代详情")
                                iterations = sample.get('iterations', [])
                                iteration_scores = sample.get('iteration_scores', [])
                                best_iteration_idx = sample.get('best_iteration_idx', -1)
                                best_score = sample.get('best_score', sample.get('score', 0))
                                final_score = sample.get('final_score', 0)
                                    
                                if iterations:
                                    # 显示关键信息
                                    col_iter1, col_iter2, col_iter3 = st.columns(3)
                                    with col_iter1:
                                        st.metric("总迭代次数", len(iterations))
                                    with col_iter2:
                                        st.metric("最高分", f"{best_score:.4f}", 
                                                 help=f"出现在第 {best_iteration_idx + 1} 次迭代" if best_iteration_idx >= 0 else None)
                                    with col_iter3:
                                        delta = final_score - best_score if final_score != best_score else None
                                        st.metric("最终分数", f"{final_score:.4f}", 
                                                 delta=f"{delta:.4f}" if delta else None,
                                                 help="最后一次迭代的分数")
                                        
                                    # 显示每次迭代的得分
                                    if iteration_scores:
                                        st.markdown("**各迭代得分**:")
                                        score_cols = st.columns(min(len(iteration_scores), 6))
                                        for idx, score in enumerate(iteration_scores):
                                            with score_cols[idx % 6]:
                                                # 标记最高分
                                                is_best = (idx == best_iteration_idx)
                                                label = f"{'🏆 ' if is_best else ''}迭代 {idx + 1}"
                                                st.metric(label, f"{score:.4f}")
                                        
                                    # 显示每次迭代的详细信息
                                    for iter_idx, iteration in enumerate(iterations):
                                        is_best = (iter_idx == best_iteration_idx)
                                        iter_score = iteration_scores[iter_idx] if iter_idx < len(iteration_scores) else 0
                                        title = f"{'🏆 ' if is_best else ''}迭代 {iter_idx + 1} 详情 (得分: {iter_score:.4f})"
                                        # 默认展开最佳迭代
                                        with st.expander(title, expanded=is_best):
                                            # 显示该迭代的对话历史
                                            messages = iteration.get('messages', [])
                                            if messages:
                                                st.markdown(f"**对话历史 ({len(messages)} 条消息)**:")
                                                for msg_idx, msg in enumerate(messages):
                                                    role = msg.get('role', 'unknown')
                                                    content = msg.get('content', '')
                                                        
                                                    # 根据角色设置样式
                                                    if role == 'user':
                                                        st.markdown(f"**👤 User (消息 {msg_idx + 1}):**")
                                                    elif role == 'assistant':
                                                        st.markdown(f"**🤖 Assistant (消息 {msg_idx + 1}):**")
                                                    elif role == 'tool':
                                                        st.markdown(f"**🔧 Tool (消息 {msg_idx + 1}):**")
                                                    else:
                                                        st.markdown(f"**📝 {role.capitalize()} (消息 {msg_idx + 1}):**")
                                                        
                                                    # 显示内容
                                                    if content:
                                                        st.text_area(
                                                            f"iter_{iter_idx}_msg_{msg_idx}",
                                                            content,
                                                            height=150,
                                                            key=f"analysis_iter_{i}_{iter_idx}_{msg_idx}",
                                                            label_visibility="collapsed"
                                                        )
                                                        
                                                    # 显示 tool_calls（如果有）
                                                    if msg.get('tool_calls'):
                                                        st.markdown("**🔧 工具调用:**")
                                                        for tool_call in msg['tool_calls']:
                                                            if isinstance(tool_call, dict):
                                                                tool_name = tool_call.get('function', {}).get('name', 'unknown')
                                                                tool_args = tool_call.get('function', {}).get('arguments', '{}')
                                                                st.code(f"Function: {tool_name}\nArguments: {tool_args}", language="json")
                                                        
                                                    st.markdown("---")
                                                
                                            # 显示该迭代的统计信息
                                            turn_record = iteration.get('turn_record', {})
                                            if turn_record:
                                                st.markdown("**轮次统计**:")
                                                for turn_key, turn_data in turn_record.items():
                                                    st.write(f"- {turn_key}: {turn_data}")
                                else:
                                    st.info("无迭代信息")
                            else:
                                # 标准格式：显示完整对话历史
                                st.markdown("### 💬 完整对话历史")
                                if sample.get('messages'):
                                    with st.expander(f"查看对话历史 ({len(sample['messages'])} 条消息)", expanded=False):
                                        for idx, msg in enumerate(sample['messages']):
                                            role = msg.get('role', 'unknown')
                                            content = msg.get('content', '')
                                            
                                            # 根据角色设置样式
                                            if role == 'user':
                                                st.markdown(f"**👤 User (消息 {idx + 1}):**")
                                            elif role == 'assistant':
                                                st.markdown(f"**🤖 Assistant (消息 {idx + 1}):**")
                                            elif role == 'system':
                                                st.markdown(f"**⚙️ System (消息 {idx + 1}):**")
                                            elif role == 'tool':
                                                st.markdown(f"**🔧 Tool (消息 {idx + 1}):**")
                                            else:
                                                st.markdown(f"**📝 {role.capitalize()} (消息 {idx + 1}):**")
                                            
                                            # 显示内容
                                            if content:
                                                st.text_area(
                                                    f"content_{idx}",
                                                    content,
                                                    height=100,
                                                    key=f"analysis_sample_{i}_{idx}",
                                                    label_visibility="collapsed"
                                                )
                                            
                                            # 显示 tool_calls（如果有）
                                            if msg.get('tool_calls'):
                                                st.markdown("**🔧 工具调用:**")
                                                for tool_call in msg['tool_calls']:
                                                    tool_name = tool_call.get('function', {}).get('name', 'unknown')
                                                    tool_args = tool_call.get('function', {}).get('arguments', '{}')
                                                    st.code(f"Function: {tool_name}\nArguments: {tool_args}", language="json")
                                            
                                            st.markdown("---")
                                else:
                                    st.info("无对话历史")
                            
                            # 原始数据（供调试使用）
                            with st.expander("🔍 查看原始 JSON 数据", expanded=False):
                                st.json(sample)
                else:
                    st.info("点击 '加载样本' 按钮查看样本详情")
                
            with tab3:
                st.subheader("可视化分析")
                
                # 数据源对比（基于统计信息）
                data_sources = response.get("data_sources", [])
                if data_sources:
                    st.markdown("#### 数据源对比")
                    ds_fig = create_data_source_comparison_chart(data_sources)
                    if ds_fig:
                        st.plotly_chart(ds_fig, width='stretch')
                
                # 显示提示：详细的样本级可视化需要加载样本
                st.info("💡 提示：样本级别的详细可视化（分数分布、Token使用、轮次分布等）需要在 '样本列表' 标签页中加载样本后才能查看。")
                
                with tab4:
                    st.subheader("错误分析")
                    
                    errors = response.get("errors", [])
                    if errors:
                        st.markdown(f"**共 {len(errors)} 个错误**")
                        
                        # 错误类型统计
                        error_types = {}
                        for error in errors:
                            error_msg = error.get("error", "Unknown")[:50]
                            error_types[error_msg] = error_types.get(error_msg, 0) + 1
                        
                        st.markdown("#### 错误类型统计")
                        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                            st.text(f"• {error_type}... ({count} 次)")
                        
                        st.markdown("---")
                        st.markdown("#### 详细错误列表")
                        
                        for i, error in enumerate(errors[:50]):  # 限制显示前50个
                            with st.expander(
                                f"❌ [{error['data_source']}] Sample #{error['sample_id']}: {error['error'][:80]}...",
                                expanded=False
                            ):
                                st.text(f"数据源: {error['data_source']}")
                                st.text(f"生成器: {error.get('generator_name', 'N/A')}")
                                st.text(f"样本ID: {error['sample_id']}")
                                st.text(f"错误信息:")
                                st.code(error['error'])
                    else:
                        st.success("✅ 没有错误！所有样本都成功完成。")
    
    elif not file_path:
        st.info("👆 请输入评测结果文件路径或从快速选择中选择文件")

