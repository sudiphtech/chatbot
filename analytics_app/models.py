from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100)
    roll_number = models.CharField(max_length=20, unique=True)
    attendance_pct = models.FloatField()
    classes_recent = models.IntegerField()
    late_count = models.IntegerField()
    assignment_rate = models.FloatField()
    engagement = models.FloatField()

class AttendanceRecord(models.Model):
    student = models.ForeignKey(Student, related_name='attendance', on_delete=models.CASCADE)
    date = models.DateField()
    present = models.BooleanField()
