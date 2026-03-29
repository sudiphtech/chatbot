import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PRESENT_STATUSES = {"present", "late", "left early", "excused"}


def _build_training_frame(app_dir):
    students_path = os.path.join(app_dir, "students.csv")
    attendance_path = os.path.join(app_dir, "final_attendance_dataset.csv")

    if not (os.path.isfile(students_path) and os.path.isfile(attendance_path)):
        return pd.DataFrame(
            {
                "attendance_pct": [90, 60, 70, 85, 50, 78, 65, 88, 72, 68],
                "classes_recent": [8, 3, 4, 7, 2, 6, 3, 8, 5, 4],
                "late_count": [1, 5, 3, 1, 6, 2, 4, 1, 3, 4],
            }
        )

    students_df = pd.read_csv(students_path).rename(columns={"Student_ID": "roll_number"})
    attendance_df = pd.read_csv(attendance_path).rename(
        columns={
            "Student_ID": "roll_number",
            "Attendance_Status": "attendance_status",
        }
    )
    attendance_df["attendance_status"] = (
        attendance_df["attendance_status"].astype(str).str.strip().str.lower()
    )
    attendance_df["present"] = attendance_df["attendance_status"].isin(PRESENT_STATUSES)

    summary = (
        attendance_df.groupby("roll_number")
        .agg(
            classes_recent=("attendance_status", "count"),
            present_days=("present", "sum"),
            late_count=("attendance_status", lambda values: (values == "late").sum()),
        )
        .reset_index()
    )

    df = students_df[["roll_number"]].merge(summary, on="roll_number", how="left").fillna(0)
    df["classes_recent"] = df["classes_recent"].astype(int)
    df["present_days"] = df["present_days"].astype(int)
    df["late_count"] = df["late_count"].astype(int)
    df["attendance_pct"] = df.apply(
        lambda row: round((row["present_days"] / row["classes_recent"]) * 100, 2)
        if row["classes_recent"]
        else 0.0,
        axis=1,
    )
    return df[["attendance_pct", "classes_recent", "late_count"]]


def train_model():
    app_dir = os.path.dirname(__file__)
    df = _build_training_frame(app_dir)
    df["at_risk"] = (df["attendance_pct"] < 75).astype(int)

    features = ["attendance_pct", "classes_recent", "late_count"]
    X = df[features]
    y = df["at_risk"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    model_path = os.path.join(app_dir, "model.pkl")
    joblib.dump(model, model_path)
    print("Model trained and saved.")


if __name__ == "__main__":
    train_model()
