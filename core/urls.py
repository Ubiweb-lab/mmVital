from django.urls import path
from . import views
from .views import RadarView

app_name = "core"

urlpatterns = [
    # path("index/", views.index, name='index'),
    # path("fetch_data/", views.fetch_data, name='fetch_data'),
    # path("fetch_hr_rr/", views.fetch_hr_rr, name='fetch_hr_rr'),

    path("", RadarView.as_view(), name='monitor'),
    path("refresh_hr_and_rr/", RadarView.as_view(), name='refresh_hr_and_rr'),
]