import json
import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import AttendanceRecord, Student


PRESENT_STATUSES = {"present", "late", "left early", "excused"}
ABSENT_STATUSES = {"absent", "absnt"}


def _project_csv_path(filename):
    return os.path.join(settings.BASE_DIR, "analytics_app", filename)


def _normalize_status(value):
    return str(value).strip().lower()


def _load_students_csv():
    path = _project_csv_path("students.csv")
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Student_ID": "roll_number",
            "Full_Name": "name",
        }
    )
    df["roll_number"] = df["roll_number"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    return df[["roll_number", "name"]]


def _load_attendance_csv():
    path = _project_csv_path("final_attendance_dataset.csv")
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Student_ID": "roll_number",
            "Date": "date",
            "Attendance_Status": "attendance_status",
        }
    )
    df["roll_number"] = df["roll_number"].astype(str).str.strip()
    df["attendance_status"] = df["attendance_status"].map(_normalize_status)
    df["present"] = df["attendance_status"].isin(PRESENT_STATUSES)
    df["absent"] = df["attendance_status"].isin(ABSENT_STATUSES)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["roll_number", "date", "attendance_status", "present", "absent"]]


def _dashboard_stats_from_csv():
    students_df = _load_students_csv()
    attendance_df = _load_attendance_csv()

    attendance_summary = (
        attendance_df.groupby("roll_number")
        .agg(
            total_days=("date", "count"),
            present_days=("present", "sum"),
        )
        .reset_index()
    )

    merged = students_df.merge(attendance_summary, on="roll_number", how="left").fillna(
        {"total_days": 0, "present_days": 0}
    )
    merged["total_days"] = merged["total_days"].astype(int)
    merged["present_days"] = merged["present_days"].astype(int)
    merged["attendance_percentage"] = np.where(
        merged["total_days"] > 0,
        (merged["present_days"] / merged["total_days"] * 100).round(2),
        0.0,
    )

    return merged.to_dict("records")


def _build_feature_summary(attendance_df):
    summary = (
        attendance_df.groupby("roll_number")
        .agg(
            total_days=("date", "count"),
            present_days=("present", "sum"),
            late_count=("attendance_status", lambda values: (values == "late").sum()),
        )
        .reset_index()
    )
    return {
        row["roll_number"]: {
            "total_days": int(row["total_days"]),
            "present_days": int(row["present_days"]),
            "late_count": int(row["late_count"]),
        }
        for _, row in summary.iterrows()
    }


def _student_defaults(row, feature_summary):
    roll_number = str(row["roll_number"]).strip()
    stats = feature_summary.get(
        roll_number,
        {"total_days": 0, "present_days": 0, "late_count": 0},
    )
    total_days = stats["total_days"]
    present_days = stats["present_days"]
    attendance_pct = round((present_days / total_days) * 100, 2) if total_days else 0.0

    return {
        "name": str(row["name"]).strip(),
        "attendance_pct": attendance_pct,
        "classes_recent": total_days,
        "late_count": stats["late_count"],
        "assignment_rate": 0.0,
        "engagement": 0.0,
    }


def _find_student_features_by_name(student_name):
    students_df = _load_students_csv()
    attendance_df = _load_attendance_csv()
    feature_summary = _build_feature_summary(attendance_df)

    lookup_name = student_name.strip().lower()
    if not lookup_name:
        raise ValueError("Please enter a student name.")

    exact_matches = students_df[students_df["name"].str.lower() == lookup_name]
    if exact_matches.empty:
        partial_matches = students_df[students_df["name"].str.lower().str.contains(lookup_name)]
    else:
        partial_matches = exact_matches

    if partial_matches.empty:
        raise ValueError(f'No student found for name "{student_name}".')

    if len(partial_matches) > 1:
        names = ", ".join(
            f'{row["name"]} ({row["roll_number"]})'
            for _, row in partial_matches.head(5).iterrows()
        )
        raise ValueError(
            "Multiple students matched that name. Please be more specific: "
            f"{names}"
        )

    student = partial_matches.iloc[0]
    stats = feature_summary.get(
        student["roll_number"],
        {"total_days": 0, "present_days": 0, "late_count": 0},
    )
    total_days = stats["total_days"]
    present_days = stats["present_days"]
    attendance_pct = round((present_days / total_days) * 100, 2) if total_days else 0.0

    return {
        "name": student["name"],
        "roll_number": student["roll_number"],
        "attendance_pct": attendance_pct,
        "classes_recent": total_days,
        "late_count": stats["late_count"],
    }


@csrf_exempt
def predict_risk(request):
    model_path = os.path.join(settings.BASE_DIR, "analytics_app", "model.pkl")
    model = joblib.load(model_path)
    if request.method == "POST":
        data = json.loads(request.body)
        features = np.array(
            [[data["attendance_pct"], data["classes_recent"], data["late_count"]]]
        )
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result = "At Risk" if prediction == 1 else "Safe"
        return JsonResponse(
            {
                "prediction": result,
                "risk_probability": float(probability),
            }
        )


def dashboard(request):
    query = request.GET.get("q", "").strip()
    page_number = request.GET.get("page", 1)

    try:
        stats = _dashboard_stats_from_csv()
    except Exception:
        students = Student.objects.all()
        stats = []
        for student in students:
            records = student.attendance.order_by("date")
            total = records.count()
            present = records.filter(present=True).count()
            percentage = round((present / total * 100), 2) if total > 0 else 0.0
            stats.append(
                {
                    "name": student.name,
                    "roll_number": student.roll_number,
                    "attendance_percentage": percentage,
                    "total_days": total,
                    "present_days": present,
                }
            )

    if query:
        query_lower = query.lower()
        stats = [
            student
            for student in stats
            if query_lower in student["name"].lower()
            or query_lower in student["roll_number"].lower()
        ]

    paginator = Paginator(stats, 100)
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "analytics_app/dashboard.html",
        {
            "students": page_obj.object_list,
            "page_obj": page_obj,
            "query": query,
            "total_students": len(stats),
        },
    )


def import_students(request):
    try:
        students_df = _load_students_csv()
        attendance_df = _load_attendance_csv()
    except FileNotFoundError as exc:
        return JsonResponse({"error": "CSV file not found", "path": str(exc)}, status=404)

    feature_summary = _build_feature_summary(attendance_df)
    imported = 0
    for _, row in students_df.iterrows():
        _, created = Student.objects.update_or_create(
            roll_number=row["roll_number"],
            defaults=_student_defaults(row, feature_summary),
        )
        if created:
            imported += 1

    return JsonResponse({"students_imported": imported, "students_total": len(students_df)})


def predict_form(request):
    model_path = os.path.join(settings.BASE_DIR, "analytics_app", "model.pkl")
    model = joblib.load(model_path)
    context = {
        "student_name": "",
        "prediction": None,
        "risk_probability": None,
        "error": None,
        "selected_student": None,
        "used_features": None,
    }
    if request.method == "POST":
        try:
            student_name = request.POST.get("student_name", "").strip()
            context["student_name"] = student_name
            student_features = _find_student_features_by_name(student_name)
            features = np.array(
                [[
                    student_features["attendance_pct"],
                    student_features["classes_recent"],
                    student_features["late_count"],
                ]]
            )
            prediction = model.predict(features)[0]
            probability = float(model.predict_proba(features)[0][1])
            context["prediction"] = "At Risk" if prediction == 1 else "Safe"
            context["risk_probability"] = probability
            context["selected_student"] = {
                "name": student_features["name"],
                "roll_number": student_features["roll_number"],
            }
            context["used_features"] = {
                "attendance_pct": student_features["attendance_pct"],
                "classes_recent": student_features["classes_recent"],
                "late_count": student_features["late_count"],
            }
        except Exception as exc:
            context["error"] = str(exc)
    return render(request, "analytics_app/predict_form.html", context)


def import_csv(request):
    return import_students(request)


def import_attendance_csv(request):
    try:
        students_df = _load_students_csv()
        attendance_df = _load_attendance_csv()
    except FileNotFoundError as exc:
        return JsonResponse({"error": "CSV file not found", "path": str(exc)}, status=404)

    student_lookup = {}
    for _, row in students_df.iterrows():
        student = Student.objects.filter(roll_number=row["roll_number"]).first()
        if student:
            student_lookup[row["roll_number"]] = student

    imported = 0
    for _, row in attendance_df.iterrows():
        student = student_lookup.get(row["roll_number"])
        if not student or pd.isna(row["date"]):
            continue

        _, created = AttendanceRecord.objects.update_or_create(
            student=student,
            date=row["date"].date(),
            defaults={"present": bool(row["present"])},
        )
        if created:
            imported += 1

    return JsonResponse(
        {
            "attendance_imported": imported,
            "attendance_total": len(attendance_df),
        }
    )


def import_all(request):
    results = {
        "students_csv": {"imported": 0, "total": 0, "error": None},
        "attendance_csv": {"imported": 0, "total": 0, "error": None},
        "model_trained": False,
        "error": None,
    }

    try:
        students_df = _load_students_csv()
        attendance_df = _load_attendance_csv()
        results["students_csv"]["total"] = len(students_df)
        results["attendance_csv"]["total"] = len(attendance_df)
    except Exception as exc:
        results["error"] = str(exc)
        return JsonResponse(results, status=400)

    feature_summary = _build_feature_summary(attendance_df)
    imported_students = 0
    for _, row in students_df.iterrows():
        _, created = Student.objects.update_or_create(
            roll_number=row["roll_number"],
            defaults=_student_defaults(row, feature_summary),
        )
        if created:
            imported_students += 1
    results["students_csv"]["imported"] = imported_students

    student_lookup = {
        student.roll_number: student for student in Student.objects.filter(
            roll_number__in=students_df["roll_number"].tolist()
        )
    }
    imported_attendance = 0
    for _, row in attendance_df.iterrows():
        student = student_lookup.get(row["roll_number"])
        if not student or pd.isna(row["date"]):
            continue

        _, created = AttendanceRecord.objects.update_or_create(
            student=student,
            date=row["date"].date(),
            defaults={"present": bool(row["present"])},
        )
        if created:
            imported_attendance += 1
    results["attendance_csv"]["imported"] = imported_attendance

    try:
        from .train_model import train_model

        train_model()
        results["model_trained"] = True
    except Exception as exc:
        results["error"] = f"Model training failed: {exc}"

    return JsonResponse(results)
