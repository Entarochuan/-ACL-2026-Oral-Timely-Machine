import os

judge_model_infos = dict(
    model_name=os.environ.get("TIMELY_JUDGE_MODEL", "deepseekv3-2"),
    model_url=os.environ.get("TIMELY_JUDGE_BASE_URL", "http://localhost:8000/v1"),
    model_api_key=os.environ.get("TIMELY_JUDGE_API_KEY", "EMPTY"),
)
