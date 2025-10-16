from attrs import frozen
from pathlib import Path

_project_root = Path(__file__).parent.parent
_data_root: Path = _project_root / "datasets"
_results_root: Path = _project_root / "results"

@frozen
class Data:
    diagnosis: Path = _data_root / "diagnosis.data"
    balance_scale: Path = _data_root / "balance-scale.data"
    banknote: Path = _data_root / "data_banknote_authentication.txt"
    transfusion: Path = _data_root / "transfusion.data"
    car: Path = _data_root / "car.data"
    kr_vs_kp: Path = _data_root / "kr-vs-kp.data"
    pop_failures: Path = _data_root / "pop_failures.data"
    house_votes_84: Path = _data_root / "house-votes-84.data"
    glass: Path = _data_root / "glass.data"
    hayes_roth_train: Path = _data_root / "hayes-roth.data"
    hayes_roth_test: Path = _data_root / "hayes-roth.test"
    segmentation_train: Path = _data_root / "segmentation.data"
    segmentation_test: Path = _data_root / "segmentation.test"
    ionosphere: Path = _data_root / "ionosphere.data"
    monks: dict[str, Path] = {
        f"{problem_number}.{t}": _data_root / f"monks-{problem_number}.{t}"
        for problem_number in (1, 2, 3)
        for t in ['train', 'test']
    }
    parkinsons: Path = _data_root / "parkinsons.data"
    soybean: Path = _data_root / "soybean-small.data"
    tic_tac_toe: Path = _data_root / "tic-tac-toe.data"

@frozen
class Results:
    comprehensive: Path = _results_root / 'comprehensive.csv'
    mip_comparison: Path = _results_root / 'mip_comparison.csv'

@frozen
class ProjectPaths:
    data: Data = Data()
    results: Results = Results()

proj_paths = ProjectPaths()
