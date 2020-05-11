from rest_framework import serializers
from .models import Story
from django.contrib.auth import get_user_model
from accounts.serializers import UserSerializer

class StorySerializer(serializers.ModelSerializer):
    create_at = serializers.DateTimeField(format="%Y.%m.%d %H:%M")
    user = UserSerializer()
    class Meta:
        model = Story
        fields = '__all__'
