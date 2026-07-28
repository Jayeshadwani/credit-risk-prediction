from pathlib import Path
from time import perf_counter

from app.report_generator import (
    generate_underwriting_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CASES_DIR = PROJECT_ROOT / "demo_cases"


def get_demo_case_ids() -> list[str]:
    return sorted(
        path.name
        for path in DEMO_CASES_DIR.iterdir()
        if path.is_dir()
        and path.name.startswith("DEMO-")
    )


def main() -> None:
    case_ids = get_demo_case_ids()

    successful_cases: list[str] = []
    failed_cases: list[dict[str, str]] = []

    start_time = perf_counter()

    for case_id in case_ids:
        print(f"\nGenerating report for {case_id}")

        case_start = perf_counter()

        try:
            report = generate_underwriting_report(
                case_id=case_id,
                save_output=True,
            )

            elapsed = perf_counter() - case_start

            successful_cases.append(case_id)

            print(
                f"Completed {case_id} in "
                f"{elapsed:.2f} seconds"
            )

            print(
                "Recommendation:",
                report.recommendation.recommendation,
            )

        except Exception as error:
            failed_cases.append({
                "case_id": case_id,
                "error": str(error),
            })

            print(f"Failed {case_id}: {error}")

    total_time = perf_counter() - start_time

    print("\nBatch generation complete")
    print(f"Total cases: {len(case_ids)}")
    print(f"Successful: {len(successful_cases)}")
    print(f"Failed: {len(failed_cases)}")
    print(f"Total time: {total_time:.2f} seconds")

    if failed_cases:
        print("\nFailed cases:")

        for failure in failed_cases:
            print(
                failure["case_id"],
                "->",
                failure["error"],
            )

        raise RuntimeError(
            f"{len(failed_cases)} report generations failed."
        )


if __name__ == "__main__":
    main()