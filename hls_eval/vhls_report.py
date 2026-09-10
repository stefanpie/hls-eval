import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

T_unwrap = TypeVar("T_unwrap")


def unwrap(value: T_unwrap | None) -> T_unwrap:
    if value is None:
        raise ValueError("Value is None")
    return value


@dataclass
class DesignHLSSynthData:
    clock_period: float

    latency_best_cycles: int | None
    latency_best_seconds: float | None
    latency_average_cycles: int | None
    latency_average_seconds: float | None
    latency_worst_cycles: int | None
    latency_worst_seconds: float | None

    resources_lut_used: int
    resources_ff_used: int
    resources_dsp_used: int
    resources_bram_used: int
    resources_uram_used: int

    resources_lut_total: int
    resources_ff_total: int
    resources_dsp_total: int
    resources_bram_total: int
    resources_uram_total: int

    resources_lut_fraction_used: float
    resources_ff_fraction_used: float
    resources_dsp_fraction_used: float
    resources_bram_fraction_used: float
    resources_uram_fraction_used: float

    @classmethod
    def parse_from_synth_report_file(cls, fp: Path) -> "DesignHLSSynthData":
        tree = ET.parse(fp)
        root = tree.getroot()

        # Gather latency data
        performance_estimates = root.find("PerformanceEstimates")

        summary_of_timing_analysis = performance_estimates.find(  # type: ignore
            "SummaryOfTimingAnalysis"
        )

        clock_units = str(unwrap(summary_of_timing_analysis.find("unit")).text)  # type: ignore
        clock_period = float(
            unwrap(summary_of_timing_analysis.find("EstimatedClockPeriod")).text  # type: ignore
        )

        unit_scaler = 1.0
        if clock_units == "ns":
            unit_scaler = 1e-9
        elif clock_units == "us":
            unit_scaler = 1e-6
        elif clock_units == "ms":
            unit_scaler = 1e-3
        else:
            raise NotImplementedError(f"Unknown clock unit: {clock_units}")

        clock_period_t = clock_period * unit_scaler

        summary_of_overall_latency = performance_estimates.find(  # type: ignore
            "SummaryOfOverallLatency"
        )
        latency_data = {}
        # fmt: off
        best_case_latency = summary_of_overall_latency.find("Best-caseLatency").text # type: ignore
        try:
            latency_data["best_case_latency"] = int(best_case_latency) # type: ignore
        except ValueError:
            latency_data["best_case_latency"] = None # type: ignore
        average_case_latency = summary_of_overall_latency.find("Average-caseLatency").text # type: ignore
        try:
            latency_data["average_case_latency"] = int(average_case_latency) # type: ignore
        except ValueError:
            latency_data["average_case_latency"] = None # type: ignore
        worst_case_latency = summary_of_overall_latency.find("Worst-caseLatency").text # type: ignore
        try:
            latency_data["worst_case_latency"] = int(worst_case_latency) # type: ignore
        except ValueError:
            latency_data["worst_case_latency"] = None # type: ignore

        if latency_data["best_case_latency"] is not None:
            latency_data["best_case_latency_t"] = latency_data["best_case_latency"] * clock_period_t # type: ignore
        else:
            latency_data["best_case_latency_t"] = None
        if latency_data["average_case_latency"] is not None:
            latency_data["average_case_latency_t"] = latency_data["average_case_latency"] * clock_period_t# type: ignore
        else:
            latency_data["average_case_latency_t"] = None
        if latency_data["worst_case_latency"] is not None:
            latency_data["worst_case_latency_t"] = latency_data["worst_case_latency"] * clock_period_t# type: ignore
        else:
            latency_data["worst_case_latency_t"] = None
        # fmt: on

        # Gather resource data
        area_estimates = root.find("AreaEstimates")
        resource_data = {}  # type: ignore
        # fmt: off
        resource_data["used_abs"] = {}
        resource_data["used_abs"]["BRAM_18K"] = int(area_estimates.find("Resources").find("BRAM_18K").text)# type: ignore
        resource_data["used_abs"]["DSP"] = int(area_estimates.find("Resources").find("DSP").text)# type: ignore
        resource_data["used_abs"]["FF"] = int(area_estimates.find("Resources").find("FF").text)# type: ignore
        resource_data["used_abs"]["LUT"] = int(area_estimates.find("Resources").find("LUT").text)# type: ignore
        resource_data["used_abs"]["URAM"] = int(area_estimates.find("Resources").find("URAM").text)# type: ignore

        resource_data["available_abs"] = {}
        resource_data["available_abs"]["BRAM_18K"] = int(area_estimates.find("AvailableResources").find("BRAM_18K").text)# type: ignore
        resource_data["available_abs"]["DSP"] = int(area_estimates.find("AvailableResources").find("DSP").text)# type: ignore
        resource_data["available_abs"]["FF"] = int(area_estimates.find("AvailableResources").find("FF").text)# type: ignore
        resource_data["available_abs"]["LUT"] = int(area_estimates.find("AvailableResources").find("LUT").text)# type: ignore
        resource_data["available_abs"]["URAM"] = int(area_estimates.find("AvailableResources").find("URAM").text)# type: ignore

        resource_data["used_percent"] = {}
        # Checks for any cases were the available resources are 0 on a device
        # Only observed for URAM but we check all resources just in case

        if resource_data["available_abs"]["BRAM_18K"] == 0:
            assert resource_data["used_abs"]["BRAM_18K"] == 0
            resource_data["used_percent"]["BRAM_18K"] = 0.0
        else:
            resource_data["used_percent"]["BRAM_18K"] = float(resource_data["used_abs"]["BRAM_18K"] / resource_data["available_abs"]["BRAM_18K"])

        if resource_data["available_abs"]["DSP"] == 0:
            assert resource_data["used_abs"]["DSP"] == 0
            resource_data["used_percent"]["DSP"] = 0.0
        else:
            resource_data["used_percent"]["DSP"] = float(resource_data["used_abs"]["DSP"] / resource_data["available_abs"]["DSP"])

        if resource_data["available_abs"]["FF"] == 0:
            assert resource_data["used_abs"]["FF"] == 0
            resource_data["used_percent"]["FF"] = 0.0
        else:
            resource_data["used_percent"]["FF"] = float(resource_data["used_abs"]["FF"] / resource_data["available_abs"]["FF"])

        if resource_data["available_abs"]["LUT"] == 0:
            assert resource_data["used_abs"]["LUT"] == 0
            resource_data["used_percent"]["LUT"] = 0.0
        else:
            resource_data["used_percent"]["LUT"] = float(resource_data["used_abs"]["LUT"] / resource_data["available_abs"]["LUT"])

        if resource_data["available_abs"]["URAM"] == 0:
            assert resource_data["used_abs"]["URAM"] == 0
            resource_data["used_percent"]["URAM"] = 0.0
        else:
            resource_data["used_percent"]["URAM"] = float(resource_data["used_abs"]["URAM"] / resource_data["available_abs"]["URAM"])
        # fmt: on

        return cls(
            clock_period=clock_period_t,
            latency_best_cycles=latency_data["best_case_latency"],
            latency_best_seconds=latency_data["best_case_latency_t"],
            latency_average_cycles=latency_data["average_case_latency"],
            latency_average_seconds=latency_data["average_case_latency_t"],
            latency_worst_cycles=latency_data["worst_case_latency"],
            latency_worst_seconds=latency_data["worst_case_latency_t"],
            resources_lut_used=resource_data["used_abs"]["LUT"],
            resources_ff_used=resource_data["used_abs"]["FF"],
            resources_dsp_used=resource_data["used_abs"]["DSP"],
            resources_bram_used=resource_data["used_abs"]["BRAM_18K"],
            resources_uram_used=resource_data["used_abs"]["URAM"],
            resources_lut_total=resource_data["available_abs"]["LUT"],
            resources_ff_total=resource_data["available_abs"]["FF"],
            resources_dsp_total=resource_data["available_abs"]["DSP"],
            resources_bram_total=resource_data["available_abs"]["BRAM_18K"],
            resources_uram_total=resource_data["available_abs"]["URAM"],
            resources_lut_fraction_used=resource_data["used_percent"]["LUT"],
            resources_ff_fraction_used=resource_data["used_percent"]["FF"],
            resources_dsp_fraction_used=resource_data["used_percent"]["DSP"],
            resources_bram_fraction_used=resource_data["used_percent"]["BRAM_18K"],
            resources_uram_fraction_used=resource_data["used_percent"]["URAM"],
        )

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "DesignHLSSynthData":
        return cls(**d)


@dataclass
class DesignHLSCoSimData:
    # $MAX_LATENCY = "100866"
    # $MIN_LATENCY = "100866"
    # $AVER_LATENCY = "100866"
    # $MAX_THROUGHPUT = "0"
    # $MIN_THROUGHPUT = "0"
    # $AVER_THROUGHPUT = "0"
    # $TOTAL_EXECUTE_TIME = "100866"

    max_latency: int
    min_latency: int
    average_latency: int

    max_throughput: int
    min_throughput: int
    average_throughput: int

    total_execute_time: int

    @classmethod
    def parse_from_synth_report_file(cls, fp: Path) -> "DesignHLSCoSimData":
        txt = fp.read_text()
        data = {}
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("$"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("$").lower()
            value = value.strip().strip('"')
            value_int = int(value)
            data[key] = value_int

        max_latency = unwrap(data.get("max_latency"))
        min_latency = unwrap(data.get("min_latency"))
        average_latency = unwrap(data.get("aver_latency"))

        max_throughput = unwrap(data.get("max_throughput"))
        min_throughput = unwrap(data.get("min_throughput"))
        average_throughput = unwrap(data.get("aver_throughput"))

        total_execute_time = unwrap(data.get("total_execute_time"))

        return cls(
            max_latency=max_latency,
            min_latency=min_latency,
            average_latency=average_latency,
            max_throughput=max_throughput,
            min_throughput=min_throughput,
            average_throughput=average_throughput,
            total_execute_time=total_execute_time,
        )

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "DesignHLSCoSimData":
        return cls(**d)
