from django.urls import path
from .views import predict_risk, dashboard, predict_form, import_students, import_csv, import_attendance_csv, import_all

urlpatterns = [
    path('predict/', predict_risk, name='predict_risk'),
    path('dashboard/', dashboard, name='dashboard'),
    path('predict-form/', predict_form, name='predict_form'),
    path('import-students/', import_students, name='import_students'),
    path('import-csv/', import_csv, name='import_csv'),
    path('import-attendance-csv/', import_attendance_csv, name='import_attendance_csv'),
    path('import-all/', import_all, name='import_all'),
]