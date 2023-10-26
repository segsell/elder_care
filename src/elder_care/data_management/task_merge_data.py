"""Tasks for merging the data."""
import pandas as pd
import pytask
from elder_care.config import BLD
from elder_care.config import SRC


@pytask.mark.depends_on(
    {
        "data": SRC / "data" / "sharew1" / "sharew1_rel8-0-0_cv_r.dta",
    },
)
@pytask.mark.produces(BLD / "data" / "data_merged.csv")
def task_merge_waves(depends_on, produces):  # noqa: ARG001
    """Merge the data from the different waves."""
    merged = pd.DataFrame(
        {
            "name": ["Raphael", "Donatello"],
            "mask": ["red", "purple"],
            "weapon": ["sai", "bo staff"],
        },
    )
    merged.to_csv(produces, index=False)
