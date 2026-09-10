import io
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import psutil

from hls_eval.data import H_EXTENSIONS
from hls_eval.vhls_report import DesignHLSCoSimData, DesignHLSSynthData


def _compiler_defines_to_cflags(compiler_defines: list[str]) -> str:
    cflags = []
    for define in compiler_defines:
        normalized_define = define.removeprefix("-D")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:=[^{}\n\r]*)?", normalized_define):
            raise ValueError(f"Invalid compiler define: {define!r}")
        cflags.append(f"-D{normalized_define}")
    return " ".join(cflags)


def auto_find_vitis_hls_dir() -> Path | None:
    vitis_hls_bin_path_str = shutil.which("vitis_hls")
    if vitis_hls_bin_path_str is None:
        return None
    vitis_hls_dist_path = Path(vitis_hls_bin_path_str).parent.parent
    return vitis_hls_dist_path


def auto_find_vitis_hls_bin() -> Path | None:
    vitis_hls_dir = auto_find_vitis_hls_dir()
    if vitis_hls_dir is None:
        return None
    vitis_hls_bin = vitis_hls_dir / "bin" / "vitis_hls"
    if not vitis_hls_bin.exists():
        raise RuntimeError(
            f"Vitis HLS dir exists but vitis_hls bin not found: {vitis_hls_bin}"
        )
    return vitis_hls_bin


def auto_find_vitis_hls_clang_format() -> Path | None:
    vitis_hls_dist_path = auto_find_vitis_hls_dir()
    if vitis_hls_dist_path is None:
        return None
    vitis_hls_clang_format_path = (
        vitis_hls_dist_path
        / "lnx64"
        / "tools"
        / "clang-3.9-csynth"
        / "bin"
        / "clang-format"
    )
    if not vitis_hls_clang_format_path.exists():
        raise RuntimeError(
            f"Vitis HLS dir exists but clang-format not found: {vitis_hls_clang_format_path}"
        )
    return vitis_hls_clang_format_path


def auto_find_vitis_hls_lib_paths() -> list[Path] | None:
    vitis_hls_dir = auto_find_vitis_hls_dir()
    if vitis_hls_dir is None:
        return None

    lib_paths = []

    # /tools/software/xilinx/Vitis_HLS/2024.1/lib/lnx64.o/libxv_hls_llvm3.1.so
    lib_path = vitis_hls_dir / "lib" / "lnx64.o"
    if not lib_path.exists():
        raise RuntimeError(f"Vitis HLS lib path not found: {lib_path}")
    lib_paths.append(lib_path)

    # /tools/xilinx/Vitis_HLS/2024.1/lnx64/lib/csim
    lib_path = vitis_hls_dir / "lnx64" / "lib" / "csim"
    if not lib_path.exists():
        raise RuntimeError(f"Vitis HLS lib path not found: {lib_path}")
    lib_paths.append(lib_path)

    return lib_paths


def get_vitis_hls_include_dir() -> Path | None:
    # vitis_hls_dist_path = get_vitis_hls_dist_path()
    vitis_hls_dist_path = auto_find_vitis_hls_dir()
    if vitis_hls_dist_path is None:
        return None

    vitis_hls_include_dir: Path = vitis_hls_dist_path / "include"
    if not vitis_hls_include_dir.exists():
        raise RuntimeError(
            f"Vitis HLS dir exists but include dir not found: {vitis_hls_include_dir}"
        )
    return vitis_hls_include_dir


@dataclass
class ExecutionData:
    return_code: int
    stdout: str
    stderr: str
    t0: float
    t1: float
    execution_time: float
    timeout: bool


@dataclass
class ToolDataOutput:
    data_execution: ExecutionData
    data_tool: None | dict


class VitisHLSSynthTool:
    def __init__(self, vitis_hls_path: Path) -> None:
        self.vitis_hls_path = vitis_hls_path

    def run(
        self,
        build_dir: Path,
        source_files: list[Path],
        aux_files: list[Path] = [],
        build_name: str | None = None,
        build_name_prefix: str = "vitis_hls_synth_tool__",
        hls_fpga_part: str = "xczu9eg-ffvb1156-2-e",
        hls_clock_period_ns: float = 5,
        hls_top_function: str | None = None,
        hls_flow_target: str = "vivado",
        hls_unsafe_math: bool = True,
        hls_disable_auto_optimizations: bool = False,
        hls_compiler_defines: list[str] | None = None,
        timeout: float = 60.0 * 6,
    ) -> ToolDataOutput:
        if build_name is None:
            build_name = f"{build_name_prefix}{uuid.uuid4().hex}"
        else:
            build_name = f"{build_name_prefix}{build_name}"

        unique_build_dir = build_dir / build_name
        if unique_build_dir.exists():
            shutil.rmtree(unique_build_dir)
        unique_build_dir.mkdir(parents=True, exist_ok=True)

        for fp in source_files + aux_files:
            shutil.copy(fp, unique_build_dir)

        tcl_script_fp: Path = unique_build_dir / "run_hls.tcl"

        tcl_script = ""
        tcl_script += f"open_project {build_name}__proj\n"
        compiler_cflags = _compiler_defines_to_cflags(hls_compiler_defines or [])
        for fp in source_files:
            if compiler_cflags:
                tcl_script += f"add_files -cflags {{{compiler_cflags}}} {fp}\n"
            else:
                tcl_script += f"add_files {fp}\n"
        tcl_script += f"open_solution solution__synth -flow_target {hls_flow_target}\n"
        if hls_top_function is not None:
            tcl_script += f"set_top {hls_top_function}\n"
        tcl_script += f"set_part {hls_fpga_part}\n"
        tcl_script += f"create_clock -period {hls_clock_period_ns} -name clk_default\n"
        if hls_disable_auto_optimizations:
            tcl_script += "config_compile -pipeline_loops 0\n"
            tcl_script += "config_unroll -tripcount_threshold 0\n"
            tcl_script += "config_array_partition -throughput_driven off\n"
        if hls_unsafe_math:
            tcl_script += "config_compile -unsafe_math_optimizations\n"
        tcl_script += "csynth_design\n"
        tcl_script += "exit\n"

        tcl_script_fp.write_text(tcl_script)

        vitis_hls_bin = self.vitis_hls_path / "bin/vitis_hls"

        t_0 = time.monotonic()
        p = subprocess.Popen(
            [vitis_hls_bin.resolve(), "-f", tcl_script_fp.resolve()],
            cwd=unique_build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=io.DEFAULT_BUFFER_SIZE * 1024,
            text=True,
        )
        try:
            stdout, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process_id = psutil.Process(pid=p.pid)
            children = process_id.children(recursive=True)
            for child in children:
                child.terminate()
            p.terminate()
            p.communicate()

            return ToolDataOutput(
                data_execution=ExecutionData(
                    return_code=-1,
                    stdout="",
                    stderr="",
                    t0=t_0,
                    t1=time.monotonic(),
                    execution_time=timeout,
                    timeout=True,
                ),
                data_tool=None,
            )

        t_1 = time.monotonic()
        dt: float = t_1 - t_0

        if p.returncode != 0:
            return ToolDataOutput(
                data_execution=ExecutionData(
                    return_code=p.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    t0=t_0,
                    t1=t_1,
                    execution_time=dt,
                    timeout=False,
                ),
                data_tool=None,
            )

        solution_dir = unique_build_dir / f"{build_name}__proj/solution__synth"
        report_dir: Path = solution_dir / "syn" / "report"
        csynth_rpt_fp = report_dir / "csynth.xml"

        synthesis_data = DesignHLSSynthData.parse_from_synth_report_file(csynth_rpt_fp)

        return ToolDataOutput(
            data_execution=ExecutionData(
                return_code=p.returncode,
                stdout=stdout,
                stderr=stderr,
                t0=t_0,
                t1=t_1,
                execution_time=dt,
                timeout=False,
            ),
            data_tool=synthesis_data.to_dict(),
        )


class VitisHLSCSimTool:
    def __init__(self, vitis_hls_path: Path) -> None:
        self.vitis_hls_path = vitis_hls_path

    def run(
        self,
        build_dir: Path,
        source_files: list[Path],
        aux_files: list[Path] = [],
        build_name: str | None = None,
        build_name_prefix: str = "vitis_hls_csim_tool__",
        hls_fpga_part: str = "xczu9eg-ffvb1156-2-e",
        hls_clock_period_ns: float = 5,
        hls_top_function: str | None = None,
        hls_flow_target: str = "vivado",
        warn_all: bool = False,
        timeout: float = 60.0 * 2,
    ) -> tuple[ToolDataOutput, ToolDataOutput | None]:
        if build_name is None:
            build_name = f"{build_name_prefix}{uuid.uuid4().hex}"
        else:
            build_name = f"{build_name_prefix}{build_name}"

        unique_build_dir = build_dir / build_name
        if unique_build_dir.exists():
            shutil.rmtree(unique_build_dir)
        unique_build_dir.mkdir(parents=True, exist_ok=True)

        for fp in source_files + aux_files:
            shutil.copy(fp, unique_build_dir)

        tcl_script_fp: Path = unique_build_dir / "run_hls.tcl"

        tcl_script = ""
        tcl_script += f"open_project {build_name}__proj\n"
        for fp in source_files:
            # tcl_script += f"add_files -tb {fp}\n"
            if warn_all:
                tcl_script += (
                    f'add_files -tb -cflags "-Wall -Wextra -Wno-unused-function" {fp}\n'
                )
            else:
                tcl_script += f"add_files -tb {fp}\n"
        for fp in aux_files:
            tcl_script += f"add_files -tb {fp}\n"
        tcl_script += f"open_solution solution__synth -flow_target {hls_flow_target}\n"
        if hls_top_function is not None:
            tcl_script += f"set_top {hls_top_function}\n"
        tcl_script += f"set_part {hls_fpga_part}\n"
        tcl_script += f"create_clock -period {hls_clock_period_ns} -name clk_default\n"
        tcl_script += "csim_design -setup\n"
        tcl_script += "exit\n"

        tcl_script_fp.write_text(tcl_script)

        vitis_hls_bin = self.vitis_hls_path / "bin/vitis_hls"

        t_0 = time.monotonic()
        p_compile = subprocess.Popen(
            [vitis_hls_bin.resolve(), "-f", tcl_script_fp.resolve()],
            cwd=unique_build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=io.DEFAULT_BUFFER_SIZE * 1024,
            text=True,
        )
        try:
            compile_stdout, compile_stderr = p_compile.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process_id = psutil.Process(pid=p_compile.pid)
            children = process_id.children(recursive=True)
            for child in children:
                child.terminate()
            p_compile.terminate()
            p_compile.communicate()

            return ToolDataOutput(
                data_execution=ExecutionData(
                    return_code=-1,
                    stdout="",
                    stderr="",
                    t0=t_0,
                    t1=time.monotonic(),
                    execution_time=timeout,
                    timeout=True,
                ),
                data_tool=None,
            ), None

        t_1 = time.monotonic()
        dt: float = t_1 - t_0

        compile_data = ToolDataOutput(
            data_execution=ExecutionData(
                return_code=p_compile.returncode,
                stdout=compile_stdout,
                stderr=compile_stderr,
                t0=t_0,
                t1=t_1,
                execution_time=dt,
                timeout=False,
            ),
            data_tool=None,
        )

        if p_compile.returncode != 0:
            return compile_data, None

        t_0 = time.monotonic()

        csim_exe_fp = (
            unique_build_dir / f"{build_name}__proj/solution__synth/csim/build/csim.exe"
        )
        p_run = subprocess.Popen(
            [csim_exe_fp.resolve()],
            cwd=unique_build_dir
            / f"{build_name}__proj"
            / "solution__synth"
            / "csim"
            / "build",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=io.DEFAULT_BUFFER_SIZE * 1024,
            text=True,
        )

        try:
            run_stdout, run_stderr = p_run.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process_id = psutil.Process(pid=p_run.pid)
            children = process_id.children(recursive=True)
            for child in children:
                child.terminate()
            p_run.terminate()
            p_run.communicate()

            return compile_data, ToolDataOutput(
                data_execution=ExecutionData(
                    return_code=-1,
                    stdout="",
                    stderr="",
                    t0=t_0,
                    t1=time.monotonic(),
                    execution_time=timeout,
                    timeout=True,
                ),
                data_tool=None,
            )

        t_1 = time.monotonic()
        dt = t_1 - t_0

        run_data = ToolDataOutput(
            data_execution=ExecutionData(
                return_code=p_run.returncode,
                stdout=run_stdout,
                stderr=run_stderr,
                t0=t_0,
                t1=t_1,
                execution_time=dt,
                timeout=False,
            ),
            data_tool=None,
        )

        return compile_data, run_data


class VitisHLSCoSimTool:
    def __init__(self, vitis_hls_path: Path) -> None:
        self.vitis_hls_path = vitis_hls_path

    def run(
        self,
        build_dir: Path,
        source_files: list[Path],
        aux_files: list[Path] = [],
        build_name: str | None = None,
        build_name_prefix: str = "vitis_hls_synth_tool__",
        hls_fpga_part: str = "xczu9eg-ffvb1156-2-e",
        hls_clock_period_ns: float = 5,
        hls_top_function: str | None = None,
        hls_flow_target: str = "vivado",
        hls_unsafe_math: bool = True,
        hls_disable_auto_optimizations: bool = False,
        hls_compiler_defines: list[str] | None = None,
        timeout: float = 60.0 * 6,
    ) -> ToolDataOutput:
        if build_name is None:
            build_name = f"{build_name_prefix}{uuid.uuid4().hex}"
        else:
            build_name = f"{build_name_prefix}{build_name}"

        unique_build_dir = build_dir / build_name
        if unique_build_dir.exists():
            shutil.rmtree(unique_build_dir)
        unique_build_dir.mkdir(parents=True, exist_ok=True)

        for fp in source_files + aux_files:
            shutil.copy(fp, unique_build_dir)

        tcl_script_fp: Path = unique_build_dir / "run_hls.tcl"

        tcl_script = ""
        tcl_script += f"open_project {build_name}__proj\n"
        compiler_cflags = _compiler_defines_to_cflags(hls_compiler_defines or [])
        for fp in source_files:
            if compiler_cflags:
                tcl_script += f"add_files -cflags {{{compiler_cflags}}} {fp}\n"
            else:
                tcl_script += f"add_files {fp}\n"
        for fp in aux_files:
            tcl_script += f"add_files -tb {fp}\n"
        tcl_script += f"open_solution solution__synth -flow_target {hls_flow_target}\n"
        if hls_top_function is not None:
            tcl_script += f"set_top {hls_top_function}\n"
        tcl_script += f"set_part {hls_fpga_part}\n"
        tcl_script += f"create_clock -period {hls_clock_period_ns} -name clk_default\n"
        if hls_disable_auto_optimizations:
            tcl_script += "config_compile -pipeline_loops 0\n"
            tcl_script += "config_unroll -tripcount_threshold 0\n"
            tcl_script += "config_array_partition -throughput_driven off\n"
        if hls_unsafe_math:
            tcl_script += "config_compile -unsafe_math_optimizations\n"
        tcl_script += "csynth_design\n"
        tcl_script += "cosim_design\n"
        tcl_script += "exit\n"

        tcl_script_fp.write_text(tcl_script)

        vitis_hls_bin = self.vitis_hls_path / "bin/vitis_hls"

        t_0 = time.monotonic()
        p = subprocess.Popen(
            [vitis_hls_bin.resolve(), "-f", tcl_script_fp.resolve()],
            cwd=unique_build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=io.DEFAULT_BUFFER_SIZE * 1024,
            text=True,
        )
        try:
            stdout, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process_id = psutil.Process(pid=p.pid)
            children = process_id.children(recursive=True)
            for child in children:
                child.terminate()
            p.terminate()
            p.communicate()

            return ToolDataOutput(
                data_execution=ExecutionData(
                    return_code=-1,
                    stdout="",
                    stderr="",
                    t0=t_0,
                    t1=time.monotonic(),
                    execution_time=timeout,
                    timeout=True,
                ),
                data_tool=None,
            )

        t_1 = time.monotonic()
        dt: float = t_1 - t_0

        if p.returncode != 0:
            return ToolDataOutput(
                data_execution=ExecutionData(
                    return_code=p.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    t0=t_0,
                    t1=t_1,
                    execution_time=dt,
                    timeout=False,
                ),
                data_tool=None,
            )

        solution_dir = unique_build_dir / f"{build_name}__proj/solution__synth"

        report_dir: Path = solution_dir / "syn" / "report"
        csynth_rpt_fp = report_dir / "csynth.xml"
        synthesis_data = DesignHLSSynthData.parse_from_synth_report_file(csynth_rpt_fp)

        cosim_report_dir = solution_dir / "sim" / "report" / "verilog"
        lat_rpt_fp = cosim_report_dir / "lat.rpt"
        cosim_data = DesignHLSCoSimData.parse_from_synth_report_file(lat_rpt_fp)

        data_tool_combined = {
            "data_synthesis": synthesis_data.to_dict(),
            "data_cosim": cosim_data.to_dict(),
        }

        return ToolDataOutput(
            data_execution=ExecutionData(
                return_code=p.returncode,
                stdout=stdout,
                stderr=stderr,
                t0=t_0,
                t1=t_1,
                execution_time=dt,
                timeout=False,
            ),
            data_tool=data_tool_combined,
        )


class CPPCompilerTool:
    def __init__(self, vitis_hls_path: Path) -> None:
        self.vitis_hls_path = vitis_hls_path

    def run(
        self,
        build_dir: Path,
        source_files: list[Path],
        aux_files: list[Path] = [],
        build_name: str | None = None,
        build_name_prefix: str = "c_compiler_tool__",
    ) -> tuple[ToolDataOutput, ToolDataOutput | None]:
        if build_name is None:
            build_name = f"{build_name_prefix}{uuid.uuid4().hex}"
        else:
            build_name = f"{build_name_prefix}{build_name}"

        unique_build_dir = build_dir / build_name
        if unique_build_dir.exists():
            shutil.rmtree(unique_build_dir)
        unique_build_dir.mkdir(parents=True, exist_ok=True)

        for fp in source_files + aux_files:
            shutil.copy(fp, unique_build_dir)

        library_paths = []
        library_paths.append(self.vitis_hls_path / "lib" / "lnx64.o")
        library_paths.append(self.vitis_hls_path / "lnx64" / "lib" / "csim")

        env_for_vitis_hls_clang = os.environ.copy()
        if "LD_LIBRARY_PATH" in env_for_vitis_hls_clang:
            for lib_path in library_paths:
                env_for_vitis_hls_clang["LD_LIBRARY_PATH"] += f":{lib_path}"
        else:
            env_for_vitis_hls_clang["LD_LIBRARY_PATH"] = ":".join(
                [str(p) for p in library_paths]
            )

        CC = self.vitis_hls_path / "lnx64/tools/clang-3.9-csynth/bin/clang++"
        CFLAGS = [
            "-std=c++14",
            "-O3",
            "-g",
            "-fPIC",
            "-fPIE",
            "-lm",
            "-Wl,--sysroot=/",
            f"-I{self.vitis_hls_path}/include",
            f"-I{self.vitis_hls_path}/include/etc",
            f"-I{self.vitis_hls_path}/include/utils",
        ]

        out_bin_fp = unique_build_dir / "out"

        source_files_no_headers = [
            fp for fp in source_files if fp.suffix not in H_EXTENSIONS
        ]

        p_call = []
        p_call.append(str(CC.resolve()))
        p_call.extend([str(fp.resolve()) for fp in source_files_no_headers])
        p_call.append("-o")
        p_call.append(str(out_bin_fp.resolve()))
        p_call.extend(CFLAGS)

        t_0 = time.monotonic()
        p = subprocess.run(
            p_call,
            capture_output=True,
            text=True,
            env=env_for_vitis_hls_clang,
            bufsize=io.DEFAULT_BUFFER_SIZE * 1024,
        )
        t_1 = time.monotonic()

        dt = t_1 - t_0

        compile_output = ToolDataOutput(
            data_execution=ExecutionData(
                return_code=p.returncode,
                stdout=p.stdout,
                stderr=p.stderr,
                t0=t_0,
                t1=t_1,
                execution_time=dt,
                timeout=False,
            ),
            data_tool=None,
        )

        if p.returncode != 0:
            return compile_output, None

        t_0 = time.monotonic()
        p = subprocess.run(
            [out_bin_fp],
            capture_output=True,
            text=True,
        )
        t_1 = time.monotonic()

        dt = t_1 - t_0

        execute_output = ToolDataOutput(
            data_execution=ExecutionData(
                return_code=p.returncode,
                stdout=p.stdout,
                stderr=p.stderr,
                execution_time=dt,
                t0=t_0,
                t1=t_1,
                timeout=False,
            ),
            data_tool=None,
        )

        return compile_output, execute_output
