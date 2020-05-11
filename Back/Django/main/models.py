from django.db import models
from django.conf import settings


class Story(models.Model):
    title = models.CharField(max_length=100)
    image = models.TextField()
    firstimage = models.CharField(max_length=100, blank=True, null=True)
    text = models.TextField()
    create_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    like_users = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name="story_like", blank=True, null=True)
    def __str__(self):
        return self.title
