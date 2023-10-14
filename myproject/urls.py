from django.contrib import admin
from django.urls import include, path
from myapp import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),
    path('about/', views.about, name='about'),
    path('fruit_prediction/', views.fruit_prediction, name='fruit_prediction'),
    path('veg_prediction/', views.veg_prediction, name='veg_prediction'),
]
