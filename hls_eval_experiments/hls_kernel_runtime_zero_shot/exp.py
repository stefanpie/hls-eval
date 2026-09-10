import logging
from pathlib import Path

from dotenv import dotenv_values

from hls_eval.data import BenchmarkCase, find_benchmark_case_dirs
from hls_eval.eval_kernel_runtime.eval_kernel_runtime import (
    HLSKernelRuntimeZeroShotEvaluator,
)
from hls_eval.llms import build_model_remote_openrouter
from hls_eval.tools import VitisHLSCoSimTool, VitisHLSCSimTool, auto_find_vitis_hls_dir
from hls_eval.utils import check_key, unwrap

EXP_NAME = "hls_kernel_runtime_zero_shot"
# MODEL_NAME = "deepseek/deepseek-v4-flash"
MODEL_NAME = "openai/gpt-oss-120b"
K_SAMPLES = 10

DIR_CURRENT = Path(__file__).resolve().parent
DIR_ROOT = DIR_CURRENT.parent.parent
DIR_HLS_EVAL_DATA = DIR_ROOT / "hls_eval_data"
DIR_CURRENT_OUTPUT_DATA = DIR_CURRENT / "output_data"
DIR_CURRENT_OUTPUT_DATA.mkdir(exist_ok=True)

LOGGER = logging.getLogger(EXP_NAME)
LOGGER.propagate = True
LOGGER.setLevel(logging.DEBUG)

API_KEY_OPENROUTER = check_key(dotenv_values(".env")["OPENROUTER_API_KEY"])


DESIGNS_TO_RUN: None | list[str] = None

# DESIGNS_TO_RUN: None | list[str] = [
#     "covariance",
# ]


if __name__ == "__main__":
    benchmark_cases = [
        BenchmarkCase(case_dir, name=case_dir.name)
        for case_dir in find_benchmark_case_dirs(DIR_HLS_EVAL_DATA)
    ]
    benchmark_cases = sorted(
        (
            case
            for case in benchmark_cases
            if "polybench__fixed__small" in case.tags_all
        ),
        key=lambda case: case.name,
    )
    if DESIGNS_TO_RUN is not None:
        benchmark_cases = [
            case for case in benchmark_cases if case.name in DESIGNS_TO_RUN
        ]

    models = [
        build_model_remote_openrouter(
            MODEL_NAME,
            api_key=API_KEY_OPENROUTER,
        )
    ]

    vitis_hls_dir = unwrap(
        auto_find_vitis_hls_dir(),
        "Vitis HLS bin not auto found",
    )
    evaluator = HLSKernelRuntimeZeroShotEvaluator(
        vitis_hls_tool_csim=VitisHLSCSimTool(vitis_hls_dir),
        vitis_hls_tool_cosim=VitisHLSCoSimTool(vitis_hls_dir),
        output_data_dir=DIR_CURRENT_OUTPUT_DATA,
        n_samples=K_SAMPLES,
        hls_disable_auto_optimizations=True,
        hls_unsafe_math=True,
    )

    evaluator.evaluate_designs(
        benchmark_cases=benchmark_cases,
        models=models,
        n_jobs=64,
        n_jobs_pool_llm=96,
        n_jobs_pool_synth=64,
        n_jobs_pool_csim=64,
        n_jobs_pool_agent=64,
    )
