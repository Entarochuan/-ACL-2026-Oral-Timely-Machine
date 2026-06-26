#!/usr/bin/env python3
"""
测试所有import路径是否正确
在启动服务前运行此脚本检查依赖
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("InternBootcamp Web Service - Import 测试")
print("="*70)
print(f"项目根目录: {project_root}")
print()

# 测试结果统计
total_tests = 0
passed_tests = 0
failed_tests = []

def test_import(module_path, description):
    """测试单个import"""
    global total_tests, passed_tests
    total_tests += 1
    try:
        __import__(module_path)
        print(f"✅ {description}")
        passed_tests += 1
        return True
    except Exception as e:
        print(f"❌ {description}: {e}")
        failed_tests.append((description, str(e)))
        return False

print("1. 测试后端核心模块")
print("-"*70)
test_import("internbootcamp.web_service.backend.main", "backend.main")
test_import("internbootcamp.web_service.backend.models.schemas", "backend.models.schemas")
test_import("internbootcamp.web_service.backend.utils.file_manager", "backend.utils.file_manager")
test_import("internbootcamp.web_service.backend.utils.result_parser", "backend.utils.result_parser")
test_import("internbootcamp.web_service.backend.utils.task_manager", "backend.utils.task_manager")
test_import("internbootcamp.web_service.backend.services.data_generation", "backend.services.data_generation")
test_import("internbootcamp.web_service.backend.services.evaluation", "backend.services.evaluation")

print()
print("2. 测试前端核心模块")
print("-"*70)
test_import("internbootcamp.web_service.frontend.utils.api_client", "frontend.utils.api_client")

# visualizations 依赖 plotly，可能会失败
if not test_import("internbootcamp.web_service.frontend.utils.visualizations", "frontend.utils.visualizations"):
    print("   提示: 需要安装 plotly: pip install plotly")

print()
print("3. 测试依赖库")
print("-"*70)
test_import("fastapi", "FastAPI")
test_import("uvicorn", "Uvicorn")
test_import("streamlit", "Streamlit")
test_import("plotly", "Plotly")
test_import("pandas", "Pandas")
test_import("requests", "Requests")

print()
print("="*70)
print("测试总结")
print("="*70)
print(f"总测试数: {total_tests}")
print(f"通过: {passed_tests}")
print(f"失败: {len(failed_tests)}")
print()

if failed_tests:
    print("失败的测试:")
    for desc, error in failed_tests:
        print(f"  - {desc}")
        if "No module named" in error:
            module_name = error.split("'")[1] if "'" in error else "unknown"
            print(f"    解决方法: pip install {module_name}")
    print()
    print("⚠️  请先安装缺失的依赖:")
    print("    pip install -r internbootcamp/web_service/requirements_frontend.txt")
else:
    print("✅ 所有测试通过！可以启动服务了。")
    print()
    print("启动命令:")
    print("  后端: cd internbootcamp/web_service && ./start_backend.sh")
    print("  前端: cd internbootcamp/web_service && ./start_frontend.sh")

print("="*70)

