"""Entry point for `python -m forgex`.

Usage:
    python -m forgex              # run full training pipeline
    python -m forgex pipeline     # same as above
    python -m forgex api          # start the FastAPI server
    python -m forgex dashboard    # launch the Streamlit dashboard
    python -m forgex generate     # generate synthetic data only
"""

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"pipeline", "train", "--help", "-h"}:
        _run_pipeline()
    elif sys.argv[1] == "api":
        _run_api()
    elif sys.argv[1] == "dashboard":
        _run_dashboard()
    elif sys.argv[1] == "generate":
        _run_generate()
    elif sys.argv[1] == "test":
        _run_tests()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: python -m forgex [pipeline|api|dashboard|generate|test]")
        sys.exit(1)


def _run_pipeline() -> None:
    from forgex.pipeline import main as pipeline_main
    quick = "--quick" in sys.argv
    pipeline_main(quick=quick)


def _run_api() -> None:
    import uvicorn
    uvicorn.run(
        "forgex.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload="--reload" in sys.argv or "-r" in sys.argv,
        log_level="info",
    )


def _run_dashboard() -> None:
    import subprocess
    import sys as _sys
    cmd = [
        _sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).resolve().parent.parent.parent / "frontend" / "dashboard.py"),
        "--server.port", "8501",
        "--server.headless", "true",
    ]
    subprocess.run(cmd)


def _run_generate() -> None:
    from forgex.simulation.generator import SyntheticWorldEngine
    from forgex.config import load_settings
    engine = SyntheticWorldEngine(load_settings())
    tables, hidden, bias = engine.generate()
    data_dir = engine.settings.data_dir / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(data_dir / f"{name}.parquet", index=False)
    hidden.to_parquet(data_dir / "hidden_segments.parquet", index=False)
    bias.to_parquet(data_dir / "hidden_bias_ground_truth.parquet", index=False)
    print(f"Generated {len(tables)} tables with {len(hidden)} tenants -> {data_dir}")


def _run_tests() -> None:
    import subprocess
    import sys as _sys
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    cmd = [_sys.executable, "-m", "pytest", *args]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
