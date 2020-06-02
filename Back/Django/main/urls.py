from django.urls import path
from . import views

urlpatterns = [
    path('ImageUpload/<int:id>/', views.image_upload),
    path('model_evaluation/', views.test),
    path('mystory/', views.mystory),
    path('allstory/', views.allstory),
    path('like/<int:id>/', views.like),
    path('likestories/', views.like_stories),
    # path('mylike/', views.mylike),
    path('search/', views.search),
    path('mask_rcnn/', views.mask_rcnn),
    path('resolution_up/', views.resolution_up),    
    path('inpainting/', views.inpainting),
]
