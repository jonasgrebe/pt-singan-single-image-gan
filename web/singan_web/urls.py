from django.conf import settings
from django.urls import path, include
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('submit-training-form/', views.get_training_form, name='submit-training-form'),
    path('processing/', views.show_processing_page, name='processing'),
    path(r'^(?P<task_id>[\w-]+)/$', views.get_task_info, name='get-task-info'),
    path('generate/', views.show_generate, name='generate'),
    path('generate/random-generation', views.show_random_gen_form, name='random-generation'),
    path('generate/super-res', views.show_super_res_form, name='super-res'),
    path('generate/paint2image', views.show_paint2image_form, name='paint2image'),
    path('generate/harmonization', views.show_harmonization_form, name='harmonization'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)