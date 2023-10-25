import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import pandas as pd

from src.tampering.compare import METRICS, CompareType
from src.tampering.evaluate import evaluate
from src.tampering.predictor import TamperingClassificator

SPLIT_STRING = "___"


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.reset_index(inplace=True)
    df["id"] = (
        df["view"]
        + SPLIT_STRING
        + df["sideface_name"]
        + SPLIT_STRING
        + df["gt_keypoints"].astype(str)
    )
    return df


def create_pivot(df: pd.DataFrame) -> pd.DataFrame:
    df_pivot = df.pivot_table(
        index="id",
        columns="compare_type",
        values=METRICS,
        aggfunc=lambda x: ",".join(map(str, x)),
    )
    df_pivot.columns = [
        "score_{}_{}".format(col, method) for col, method in df_pivot.columns
    ]
    df_pivot = df_pivot.reset_index()

    df_final = pd.merge(
        df[["tampered", "tampering", "dataset_split", "gt_keypoints", "id"]],
        df_pivot,
        on="id",
    )
    df_final["tampering"].fillna("", inplace=True)
    df_final.fillna(-1, inplace=True)
    return df_final


def get_data_splits(df_input: pd.DataFrame, gt_keypoints: bool = False):
    data_gt = df_input[df_input["gt_keypoints"] == True]
    data_pred = df_input[df_input["gt_keypoints"] == False]

    if gt_keypoints:
        data_train = data_gt[data_gt["dataset_split"] == "validation"]
        data_test = data_gt[data_gt["dataset_split"] == "test"]
    else:
        data_train = data_pred[data_pred["dataset_split"] == "validation"]
        data_test = data_pred[data_pred["dataset_split"] == "test"]
    return data_train, data_test


def train_predictor(
    df_final: pd.DataFrame,
    validate: bool = True,
    gt_keypoints: bool = False,
    predictor_type: str = "simple_threshold",
) -> pd.DataFrame:
    SCORES = [n for n in df_final.columns if n.startswith("score")]
    data_train, data_test = get_data_splits(df_final, gt_keypoints=gt_keypoints)

    results_performance = []
    for compare_types in [[t] for t in CompareType.SELECTION()] + [
        CompareType.SELECTION()
    ]:
        scores = [s for s in SCORES if s.split("_")[-1] in compare_types]
        scores = [s for s in scores if "_".join(s.split("_")[1:-1]) in METRICS]
        if len(scores) == 0:
            continue
        predictor = TamperingClassificator(predictor_type)
        X_train = data_train[scores].to_numpy().astype(float)
        y_train = data_train["tampered"].to_numpy().astype(int)
        ids_train = data_train["id"].to_numpy()
        predictor.set_data(X_train, y_train, ids_train)
        predictor.feature_names = [s.replace("score_", "") for s in scores]
        if validate:
            (
                train_metrics_summary,
                val_metrics_summary,
                models,
            ) = predictor.validate_model(5)
            results_performance.append(
                {
                    "predictor": predictor_type,
                    "compare_types": ", ".join(compare_types),
                    "scores": ", ".join(
                        set(["_".join(s.split("_")[1:-1]) for s in scores])
                    ),
                    "feature_importance": {
                        name: value
                        for name, value in zip(
                            predictor.feature_names,
                            models[0].feature_importances_,
                        )
                        if value > 0
                    },
                    **val_metrics_summary,
                }
            )
        else:
            predictor.test_split_size = 0
            model, train_metrics, test_metrics = predictor.train()

            X_test = data_test[scores].to_numpy().astype(float)
            y_test = data_test["tampered"].to_numpy().astype(int)
            ids_test = data_test["id"].to_numpy()
            test_metrics = evaluate(model, X_test, y_test)
            results_performance.append(
                {
                    "predictor": predictor_type,
                    "compare_types": ", ".join(compare_types),
                    "scores": ",".join(
                        set(["_".join(s.split("_")[1:-1]) for s in scores])
                    ),
                    "feature_importance": {
                        name: value
                        for name, value in zip(
                            predictor.feature_names,
                            model.feature_importances_,
                        )
                        if value > 0
                    },
                    **test_metrics,
                }
            )

        df_results_ = pd.DataFrame(results_performance)
        return df_results_


def main():
    df = load_results(ROOT / "data" / "misc" / "simscores_validation.csv")
    df_final = create_pivot(df)
    df_results = train_predictor(df_final, validate=True, gt_keypoints=False)
    df_results.to_csv("tampering_results.csv")


if __name__ == "__main__":
    main()
