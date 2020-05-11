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
]
