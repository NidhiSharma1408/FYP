# Generated by Django 4.2 on 2023-04-07 10:05

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_rename_attendance_attendancemodel'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='attendancemodel',
            name='time',
        ),
        migrations.AddField(
            model_name='attendancemodel',
            name='timestamp',
            field=models.DateTimeField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
