from django.conf import settings
from django.urls import path, re_path
from django.conf.urls.static import static
from django.urls import include

from . import views

generate_patterns = [
    path('', views.show_generate, name='generate'),
    path('random-generation', views.show_random_gen_form, name='random-generation'),
    path('super-res', views.show_super_res_form, name='super-res'),
    path('paint2image', views.show_paint2image_form, name='paint2image'),
    path('harmonization', views.show_harmonization_form, name='harmonization'),
    path('injection', views.show_injection_form, name='injection'),
]

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^load-pretrained/(?P<model_id>[0-9]+)/$', views.load_pretrained, name='load-pretrained'),
    path('upload-training-image/', views.post_training_image, name='upload-training-image'),
    path('upload-input-image/', views.post_input_image, name='upload-input-image'),
    path('submit-training-form/', views.post_training_form, name='submit-training-form'),
    path('processing/', views.show_processing_page, name='processing'),
    path(r'^(?P<task_id>[\w-]+)/$', views.get_task_info, name='get-task-info'),
    path(r'cancel-training/^(?P<task_id>[\w-]+)/$', views.cancel_training, name='cancel-training'),
    path('generate/', include(generate_patterns)),
]


# if settings.DEBUG:
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
generate_patterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)