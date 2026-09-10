import ast
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path
from textwrap import dedent
from typing import Any

from joblib import Parallel, delayed
from llm import Response

from hls_eval.data import BenchmarkCase
from hls_eval.eval import EvalThreadPools, Evaluator, serialize_eval_data
from hls_eval.llms import (
    Model,
    TAIPromptTooLong,
    TAITimeout,
    normalize_model_name,
)
from hls_eval.prompting import (
    approx_num_tokens,
    build_input_code_prompt_xml,
    extract_code_xml_from_llm_output,
)
from hls_eval.tools import ToolDataOutput, VitisHLSCoSimTool, VitisHLSCSimTool


SAFE_STDLIB_MODULES = {
    "collections",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "math",
    "operator",
    "statistics",
}
DISALLOWED_BUILTINS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}


def _build_runtime_estimation_prompt(
    benchmark_case: BenchmarkCase,
    hls_clock_period_ns: float,
    hls_fpga_part: str,
    hls_compiler_defines: list[str],
    hls_disable_auto_optimizations: bool,
    hls_unsafe_math: bool,
) -> str:
    synthesis_parameters = {
        "tool": "Vitis HLS",
        "top_function": benchmark_case.top_fn,
        "target_part": hls_fpga_part,
        "target_clock_period_ns": hls_clock_period_ns,
        "target_clock_frequency_mhz": 1000.0 / hls_clock_period_ns,
        "compiler_defines": hls_compiler_defines,
        "automatic_loop_pipelining": not hls_disable_auto_optimizations,
        "automatic_loop_unrolling": not hls_disable_auto_optimizations,
        "automatic_array_partitioning": not hls_disable_auto_optimizations,
        "unsafe_math_optimizations": hls_unsafe_math,
    }
    prompt = dedent(
        f"""
        ## Overview
        You are an expert FPGA and high-level synthesis engineer. This task tests
        your existing knowledge of how Vitis HLS schedules C/C++ kernels. You do
        not have access to Vitis HLS or any other synthesis tool, and you must not
        attempt to invoke one.

        ## Task
        Estimate the synthesized top-level kernel's overall latency in clock
        cycles for the supplied source code, testbench, and synthesis parameters.
        Predict the schedule that Vitis HLS will construct, rather than treating
        the kernel like sequential software or merely counting arithmetic
        operations.

        ## Synthesis Behavior
        - Vitis HLS automatic loop pipelining, threshold unrolling, and
          throughput-driven array partitioning are
          {"disabled; only model these transformations when explicitly requested by source pragmas" if hls_disable_auto_optimizations else "enabled and should be modeled even without explicit source pragmas"}.
        - Explicit HLS pragmas remain active and must be modeled, including their
          interactions. For example, an explicitly pipelined outer loop can
          require an inner loop to be unrolled.
        - A PIPELINE pragma is a request, not a guarantee of II=1. Poorly
          structured HLS code can produce II violations from loop-carried
          dependencies, insufficient memory ports, or limited shared resources.
          Vitis then implements the lowest achievable II, which can substantially
          increase latency relative to an ideal II=1 pipeline.
        - Unsafe math optimizations are {"enabled" if hls_unsafe_math else "disabled"}.
          When enabled, Vitis may reassociate and balance floating-point
          expressions when dependencies permit.
        - Even without explicit pipeline pragmas, HLS schedules independent
          operations concurrently, chains operations when the clock permits,
          shares or duplicates hardware resources, and applies normal compiler
          optimizations. Do not equate one C/C++ statement with one clock cycle.

        ## Estimation Guidance
        1. Determine exact minimum and maximum loop trip counts from the kernel,
           compiler defines, and relevant testbench inputs. For triangular,
           conditional, or data-dependent loops, model the conservative
           worst-case path because the target is the report's worst-case latency.
        2. Identify every explicit PIPELINE, UNROLL, ARRAY_PARTITION, DATAFLOW,
           INLINE, and interface pragma and determine how it changes the effective
           loop structure, available parallelism, and memory bandwidth.
        3. For an explicitly pipelined loop, estimate latency using pipeline depth
           plus initiation interval times the remaining iterations. Derive the II
           from loop-carried dependencies, recurrence distance, memory-port
           demand, and shared operator resources; do not assume II=1.
        4. For an unrolled loop, account for parallel operator copies and
           simultaneous array accesses. Respect partitioning and the available
           ports of unpartitioned memories.
        5. Use operation latencies appropriate for the target FPGA, clock period,
           data type, and unsafe-math setting. In a pipeline, operation latency is
           generally pipeline depth or a recurrence constraint, not a serial cost
           paid independently for every iteration.
        6. Include loop entry/exit, pipeline fill/drain, function-call, memory
           access, and top-level control overhead where relevant, but do not add
           arbitrary software branch or instruction costs.

        Write a self-contained Python script that models your estimate and prints
        the final estimate. The script may perform intermediate calculations, but:
        - avoid imports unless necessary; if needed, import only from: math,
          statistics, fractions, decimal, itertools, functools, operator, or
          collections;
        - it must contain exactly one call to `print`;
        - its output must be exactly one positive base-10 integer and a newline;
        - it must not read files, access the network, launch processes, or invoke
          HLS/synthesis tools;
        - it must be valid Python with every variable defined before use. Check
          the script mentally for syntax and runtime errors before returning it.

        ## Output Format
        Return only one XML code element and no explanation:
        <OUTPUT_CODE name="estimate.py">
        # complete Python script
        </OUTPUT_CODE>

        ## Synthesis Parameters
        """
    ).strip()
    prompt += "\n" + json.dumps(synthesis_parameters, indent=2)
    prompt += "\n\n## Task Inputs\n"
    prompt += build_input_code_prompt_xml(
        {
            source_file.name: source_file.read_text()
            for source_file in benchmark_case.source_files
        }
    )
    prompt += "\n## Task Output\n"
    return prompt


def _validate_estimator_script(code: str) -> None:
    tree = ast.parse(code, filename="estimate.py")
    print_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name.split(".", maxsplit=1)[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".", maxsplit=1)[0]]
            )
            unsupported_modules = set(modules) - SAFE_STDLIB_MODULES
            if unsupported_modules:
                raise ValueError(
                    "Estimator imports unsupported modules: "
                    + ", ".join(sorted(unsupported_modules))
                )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                print_calls += 1
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_BUILTINS:
                raise ValueError(f"Estimator uses disallowed builtin: {node.func.id}")

    if print_calls != 1:
        raise ValueError(
            f"Estimator must contain exactly one print call; found {print_calls}"
        )


def _run_estimator_script(
    script_path: Path, timeout_seconds: float
) -> tuple[int | None, dict[str, Any], str | None]:
    t0 = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", script_path.name],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        timeout = False
    except subprocess.TimeoutExpired as error:
        t1 = time.monotonic()
        execution_data = {
            "return_code": -1,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "t0": t0,
            "t1": t1,
            "execution_time": t1 - t0,
            "timeout": True,
        }
        return None, execution_data, "Estimator script timed out"

    t1 = time.monotonic()
    execution_data = {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "t0": t0,
        "t1": t1,
        "execution_time": t1 - t0,
        "timeout": timeout,
    }
    if completed.returncode != 0:
        return (
            None,
            execution_data,
            (
                f"Estimator script exited with code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            ),
        )

    stdout = completed.stdout
    if not stdout.endswith("\n") or stdout.count("\n") != 1:
        return (
            None,
            execution_data,
            "Estimator output must be one newline-terminated line",
        )
    output = stdout.strip()
    if not output.isascii() or not output.isdecimal():
        return None, execution_data, "Estimator output must be a base-10 integer"

    estimate = int(output)
    if estimate <= 0:
        return None, execution_data, "Estimator output must be positive"
    return estimate, execution_data, None


def _serialize_tool_output(tool_output: Any) -> dict[str, Any]:
    return {
        "data_execution": {
            "return_code": tool_output.data_execution.return_code,
            "stdout": tool_output.data_execution.stdout,
            "stderr": tool_output.data_execution.stderr,
            "t0": tool_output.data_execution.t0,
            "t1": tool_output.data_execution.t1,
            "execution_time": tool_output.data_execution.execution_time,
            "timeout": tool_output.data_execution.timeout,
        },
        "data_tool": tool_output.data_tool or {},
    }


class HLSKernelRuntimeZeroShotEvaluator(Evaluator):
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_cosim: VitisHLSCoSimTool,
        output_data_dir: Path,
        n_samples: int = 1,
        temperature: float | None = None,
        hls_clock_period_ns: float = 5.0,
        hls_fpga_part: str = "xczu9eg-ffvb1156-2-e",
        hls_compiler_defines: list[str] | None = None,
        hls_disable_auto_optimizations: bool = True,
        hls_unsafe_math: bool = True,
        estimator_timeout_seconds: float = 10.0,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if hls_clock_period_ns <= 0:
            raise ValueError("hls_clock_period_ns must be positive")
        if estimator_timeout_seconds <= 0:
            raise ValueError("estimator_timeout_seconds must be positive")

        self.n_samples = n_samples
        self.temperature = temperature
        self.hls_clock_period_ns = hls_clock_period_ns
        self.hls_fpga_part = hls_fpga_part
        self.hls_compiler_defines = list(hls_compiler_defines or [])
        self.hls_disable_auto_optimizations = hls_disable_auto_optimizations
        self.hls_unsafe_math = hls_unsafe_math
        self.estimator_timeout_seconds = estimator_timeout_seconds

        super().__init__(vitis_hls_tool_csim, vitis_hls_tool_cosim, output_data_dir)

    def evaluate_design(
        self,
        benchmark_case: BenchmarkCase,
        model: Model,
        pools: EvalThreadPools,
        **kwargs,
    ) -> None:
        model_name: str = model.name
        model_name_normalized = normalize_model_name(model_name)
        benchmark_case_name = benchmark_case.name
        eval_id = f"{benchmark_case_name}__{model_name_normalized}"

        eval_dir_top = self.output_data_dir / eval_id
        if eval_dir_top.exists():
            self.logger.info(f"Removing existing top eval dir: {eval_dir_top}")
            shutil.rmtree(eval_dir_top)
        eval_dir_top.mkdir(parents=True)

        synthesis_source_files = [
            source_file
            for source_file in benchmark_case.source_files
            if source_file != benchmark_case.tb_file
        ]
        ground_truth_build_dir = eval_dir_top / "ground_truth_build"
        ground_truth_build_dir.mkdir()
        synthesis_future = pools.pool_synth.submit(
            self.vitis_hls_tool.run,
            ground_truth_build_dir,
            synthesis_source_files,
            aux_files=[benchmark_case.tb_file],
            build_name=eval_id,
            hls_top_function=benchmark_case.top_fn,
            hls_fpga_part=self.hls_fpga_part,
            hls_clock_period_ns=self.hls_clock_period_ns,
            hls_disable_auto_optimizations=self.hls_disable_auto_optimizations,
            hls_unsafe_math=self.hls_unsafe_math,
            hls_compiler_defines=self.hls_compiler_defines,
        )

        Parallel(n_jobs=self.n_samples, backend="threading")(
            delayed(self._evaluate_sample)(
                sample_idx=sample_idx,
                benchmark_case=benchmark_case,
                model=model,
                pools=pools,
                eval_id=eval_id,
                benchmark_case_name=benchmark_case_name,
                model_name=model_name,
                model_name_normalized=model_name_normalized,
                eval_dir_top=eval_dir_top,
                synthesis_future=synthesis_future,
            )
            for sample_idx in range(self.n_samples)
        )

        all_eval_data = {}
        for sample_idx in range(self.n_samples):
            sample_eval_data_path = (
                eval_dir_top / f"sample__{sample_idx}" / "single_eval_data.json"
            )
            all_eval_data[sample_idx] = json.loads(sample_eval_data_path.read_text())
        (eval_dir_top / "all_eval_data.json").write_text(
            json.dumps(all_eval_data, indent=4)
        )

    def _evaluate_sample(
        self,
        sample_idx: int,
        benchmark_case: BenchmarkCase,
        model: Model,
        pools: EvalThreadPools,
        eval_id: str,
        benchmark_case_name: str,
        model_name: str,
        model_name_normalized: str,
        eval_dir_top: Path,
        synthesis_future: "Future[ToolDataOutput]",
    ) -> None:
        eval_data: dict[str, Any] = {
            "eval_type": "hls_kernel_runtime_zero_shot",
            "eval_id": eval_id,
            "benchmark_case_name": benchmark_case_name,
            "benchmark_case_tags": benchmark_case.tags_all,
            "model_name": model_name,
            "model_name_normalized": model_name_normalized,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
            "estimated_latency_cycles": None,
            "synthesis_parameters": {
                "hls_clock_period_ns": self.hls_clock_period_ns,
                "hls_clock_frequency_mhz": 1000.0 / self.hls_clock_period_ns,
                "hls_fpga_part": self.hls_fpga_part,
                "hls_compiler_defines": self.hls_compiler_defines,
                "hls_top_function": benchmark_case.top_fn,
                "hls_disable_auto_optimizations": (
                    self.hls_disable_auto_optimizations
                ),
                "hls_unsafe_math": self.hls_unsafe_math,
            },
        }
        eval_dir = eval_dir_top / f"sample__{sample_idx}"
        eval_dir.mkdir(parents=True)

        design_dir = eval_dir / "design"
        sample_benchmark_case = benchmark_case.copy_to(design_dir)
        prompt = _build_runtime_estimation_prompt(
            sample_benchmark_case,
            self.hls_clock_period_ns,
            self.hls_fpga_part,
            self.hls_compiler_defines,
            self.hls_disable_auto_optimizations,
            self.hls_unsafe_math,
        )
        eval_data["prompt"] = prompt
        (eval_dir / "raw_llm_prompt.txt").write_text(prompt)

        llm = model.llm

        def call_model() -> tuple[Response | None, str | None, bool, bool, float, float]:
            t0 = time.monotonic()
            try:
                response = llm.prompt(
                    prompt=prompt,
                    stream=False,
                    temperature=self.temperature,
                )
                response._force()
                response_text = response.text()
                return (
                    response,
                    response_text,
                    False,
                    False,
                    t0,
                    time.monotonic(),
                )
            except TAITimeout:
                return None, None, True, False, t0, time.monotonic()
            except TAIPromptTooLong:
                return None, None, False, True, t0, time.monotonic()

        self.logger.info(
            f"[{eval_id}] Calling model with approximately "
            f"{approx_num_tokens(prompt)} prompt tokens"
        )
        llm_future = pools.pool_llm.submit(call_model)
        (
            response,
            response_text,
            model_timeout,
            prompt_too_long,
            llm_t0,
            llm_t1,
        ) = llm_future.result()
        eval_data["model_timeout"] = model_timeout
        eval_data["prompt_too_long"] = prompt_too_long
        eval_data["llm_execution_time"] = {
            "t0": llm_t0,
            "t1": llm_t1,
            "execution_time": llm_t1 - llm_t0,
        }

        if response is not None and response_text is not None:
            if response.response_json is not None:
                eval_data["response_json"] = response.response_json
            eval_data["raw_output"] = response_text
            (eval_dir / "raw_llm_output.txt").write_text(response_text)

            try:
                generated_code = extract_code_xml_from_llm_output(response_text)
                if set(generated_code) != {"estimate.py"}:
                    raise ValueError(
                        "Expected exactly one OUTPUT_CODE named estimate.py"
                    )
                estimator_code = generated_code["estimate.py"].strip() + "\n"
                eval_data["generated_code"] = {"estimate.py": estimator_code}
                _validate_estimator_script(estimator_code)
                estimator_script_path = eval_dir / "estimate.py"
                estimator_script_path.write_text(estimator_code)
                (
                    estimated_cycles,
                    execution_data,
                    estimator_error,
                ) = _run_estimator_script(
                    estimator_script_path,
                    self.estimator_timeout_seconds,
                )
                eval_data["estimator_execution"] = execution_data
                if estimator_error is None:
                    eval_data["can_parse_output"] = True
                    eval_data["estimated_latency_cycles"] = estimated_cycles
                else:
                    eval_data["can_parse_output"] = False
                    eval_data["estimator_error"] = estimator_error
            except (SyntaxError, ValueError) as error:
                eval_data["can_parse_output"] = False
                eval_data["estimator_error"] = str(error)

        cosim_output = synthesis_future.result()
        eval_data["vitis_hls_tool_out"] = _serialize_tool_output(cosim_output)
        if cosim_output.data_tool:
            data_synthesis = cosim_output.data_tool.get("data_synthesis") or {}
            data_cosim = cosim_output.data_tool.get("data_cosim") or {}
            eval_data["actual_latency_cycles__synth"] = {
                "best": data_synthesis.get("latency_best_cycles"),
                "average": data_synthesis.get("latency_average_cycles"),
                "worst": data_synthesis.get("latency_worst_cycles"),
            }
            eval_data["actual_latency_cycles__cosim"] = {
                "best": data_cosim.get("min_latency"),
                "average": data_cosim.get("average_latency"),
                "worst": data_cosim.get("max_latency"),
            }
            eval_data["target_actual_latency_cycles__synth"] = data_synthesis.get(
                "latency_worst_cycles"
            )
            eval_data["target_actual_latency_cycles__cosim"] = data_cosim.get(
                "max_latency"
            )

        serialize_eval_data(eval_id, eval_dir, eval_data)
