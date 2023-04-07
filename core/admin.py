from django.contrib import admin
from .models import *
import csv
from django.http import HttpResponse
# Register your models here.
from rangefilter.filters import DateRangeFilter
admin.site.register(FaceModel)

class ExportCsvMixin:
    def export_as_csv(self, request, queryset):

        meta = self.model._meta
        field_names = [field.name for field in meta.fields]

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename={}.csv'.format(meta)
        writer = csv.writer(response)

        writer.writerow(field_names)
        for obj in queryset:
            row = writer.writerow([getattr(obj, field) for field in field_names])

        return response

    export_as_csv.short_description = "Export Selected"

class AttendanceModelAdmin(admin.ModelAdmin, ExportCsvMixin):
    fields = ['name', 'timestamp']
    list_display = ['name', 'timestamp']
    actions = ["export_as_csv"]
    list_filter = (('timestamp', DateRangeFilter),)

admin.site.register(AttendanceModel, AttendanceModelAdmin)
