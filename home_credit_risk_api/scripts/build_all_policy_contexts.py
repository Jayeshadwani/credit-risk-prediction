from pathlib import Path

from scripts.build_applicant_policy_context import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CASES_DIR = PROJECT_ROOT / "demo_cases"


case_directories = sorted(
    path
    for path in DEMO_CASES_DIR.iterdir()
    if path.is_dir() and path.name.startswith("DEMO-")
)

successful_cases = []
failed_cases = []

for case_directory in case_directories:
    case_id = case_directory.name

    try:
        print(f"\nProcessing {case_id}")
        main(case_id)
        successful_cases.append(case_id)

    except Exception as error:
        print(f"Failed {case_id}: {error}")
        failed_cases.append({
            "case_id": case_id,
            "error": str(error),
        })


print("\nGeneration complete")
print(f"Successful: {len(successful_cases)}")
print(f"Failed: {len(failed_cases)}")

if failed_cases:
    print("\nFailed cases:")

    for failure in failed_cases:
        print(
            failure["case_id"],
            "-",
            failure["error"],
        )