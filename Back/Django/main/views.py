from django.shortcuts import render, get_object_or_404
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from sub3.settings import MEDIA_ROOT

from django.http import HttpResponse, JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated, AllowAny

# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms


import os, datetime, random
from PIL import Image
from django.contrib.auth import get_user_model

import prosr_test
from .models import Story
from .serializers import *
import json

from pathlib import Path
base_path = Path(__file__).parent.absolute()
from edsr_predict import predict as edsr_prediction
# from edsr_predict import downscale_by_ratio
from inpainting_predict import predict as inpainting_predict
from segmentation_predict import predict as mask_rcnn_prediction


@api_view(["POST"])
@permission_classes(
    [IsAuthenticated,]
)
def mask_rcnn(request):
    print("views.py | mask_rcnn(request)")
    file_name = request.data["img"]
    print("file_name:{}".format(file_name))
    print("MEDIA_ROOT path:{}".format(MEDIA_ROOT))
    # base_path = Path(__file__).parent.absolute()
    print(MEDIA_ROOT + "\\")
    result = mask_rcnn_prediction(file_name, MEDIA_ROOT + "\\")
    print(result)
    return JsonResponse({"masked_images": result[0], "mask": result[1]})


@api_view(["POST"])
@permission_classes(
    [IsAuthenticated,]
)
def resolution_up_prosr(request):
    print("views.py | resolution_up(request)")
    file_name = request.data["img"]
    # output_file_name = predict(file_name,MEDIA_ROOT,AI_directory_path="/home/ubuntu/s02p23c104/Back/AI",model_type=modeltype)
    output_file_name = prosr_test.change(file_name, MEDIA_ROOT)
    print("결과@@@@@@", output_file_name)
    return JsonResponse({"resolution_up": output_file_name})


@api_view(["POST"])
@permission_classes(
    [IsAuthenticated,]
)
def resolution_up_edsr(request):
    print("views.py | resolution_up(request)")
    file_name = request.data["img"]
    print("file_name:{}".format(file_name))
    print("MEDIA_ROOT path:{}".format(MEDIA_ROOT))

    # downscale_by_ratio(file_name, MEDIA_ROOT, 2)  # Before EDSR
    base_path = Path(__file__).parent.absolute()
    print("base_path : {}".format(base_path))
    model_path = (base_path / "../../AI/experiment/edsr_baseline_x2/model/model_best.pt").resolve()
    print("model_path : {} \nmodel_path type : {}".format(model_path, type(model_path)))
    output_file_name = edsr_prediction(
        images=file_name, root_path=MEDIA_ROOT, ai_directory_path=model_path
    )
    print("output_file_name:{}".format(output_file_name))
    # downscale_by_ratio(output_file_name[0], MEDIA_ROOT, 2)  # After EDSR
    return JsonResponse({'resolution_up':output_file_name})


@api_view(["POST"])
@permission_classes(
    [IsAuthenticated,]
)
def inpainting(request):
    print("inpainting시작")
    original_image = request.data["img"]
    mask = request.data["mask"]

    model_path = (base_path / "../../AI/experiment/inpainting/model/inpainting_model.pth").resolve()
    output_file_name = inpainting_predict(
        original_image, mask, MEDIA_ROOT, AI_directory_path=model_path, model_type=""
    )

    return JsonResponse({"inpainting": output_file_name})


def uploaded(f):
    name = str(datetime.datetime.now().strftime("%H%M%S")) + str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + "/" + name, ContentFile(f.read()))
    temppath = name
    return os.path.join(MEDIA_ROOT, path), name, temppath

@api_view(["POST"])
@permission_classes(
    [IsAuthenticated,]
)
def image_upload(request, id):
    file_path = ""
    image = []
    file_names = []
    text = ["abc", "def"]
    modeltype = id
    # if id == 1:
    #     modeltype = 'life'  # 수필
    # elif id == 2:
    #     modeltype = 'story'  # 소설
    # elif id == 3:
    #     modeltype = 'news'  # 뉴스
    for _file in request.FILES.getlist("images[]"):
        request.FILES["images[]"] = _file
        file_path, file_name, path = uploaded(request.FILES["images[]"])
        image.append(path)
        file_names.append(file_name)
    # text = predict(file_names,MEDIA_ROOT,AI_directory_path="/home/ubuntu/s02p23c104/Back/AI",model_type=modeltype)
    return JsonResponse({"result": "true", "text": text, "image": image})


@api_view(
    ["GET", "POST",]
)
@permission_classes(
    [IsAuthenticated,]
)
def mystory(request):
    if request.method == "POST":
        img = {}
        title = request.data["title"]
        image = request.data["img"]
        text = request.data["content"]
        for i in range(len(image)):
            img[str(i)] = image[i]
            if i == 0:
                firstimage = image[i]
        story = Story.objects.create(
            title=title, image=img, text=text, firstimage=firstimage, user=request.user
        )
        story.save()
        return JsonResponse({"result": "true", "title": title, "text": text, "image": image,})
    elif request.method == "GET":
        user = get_object_or_404(get_user_model(), username=request.user)
        story = Story.objects.filter(user_id=user.id).order_by("-create_at")
        temp = story.values()
        first_image = []
        for i in temp:
            image_string = i["image"]
            image_string = image_string.replace("'", '"')
            image_dict = json.loads(image_string)
            first_image.append(image_dict["0"])
        serializers = StorySerializer(story, many=True)
        return JsonResponse(
            {"result": "true", "data": serializers.data, "first_image": first_image}
        )


@api_view(["GET"])
@permission_classes(
    [AllowAny,]
)
def allstory(request):
    rank = {}
    storys = Story.objects.all()
    for story in storys.values("id", "like_users"):
        if story["id"] in rank and story["like_users"] != None:
            rank[story["id"]] += 1
        elif story["id"] not in rank and story["like_users"] != None:
            rank[story["id"]] = 1
        elif story["id"] not in rank and story["like_users"] == None:
            rank[story["id"]] = 0
    storys = sorted(list(rank.items()), key=lambda x: x[1], reverse=True)
    rank_story = [get_object_or_404(Story, id=story[0]) for story in storys]
    serializer = StorySerializer(rank_story, many=True)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes(
    [IsAuthenticated,]
)
def like(request, id):
    story = Story.objects.get(id=id)
    user = request.user
    if user not in story.like_users.all():
        story.like_users.add(user)
        on_like = True
    else:
        story.like_users.remove(user)
        on_like = False
    return JsonResponse(
        {"result": "true", "count_like": story.like_users.all().count(), "on_like": on_like}
    )


@api_view(["GET"])
@permission_classes(
    [IsAuthenticated,]
)
def like_stories(request):
    user = get_object_or_404(get_user_model(), username=request.user)
    stories = user.story_like.all()
    serializer = StorySerializer(stories, many=True)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes(
    [AllowAny,]
)
def search(request):
    query = request.data["keyword"]
    if query:
        story = Story.objects.filter(title__contains=query)
    serializer = StorySerializer(story, many=True)
    return Response(serializer.data)


# @api_view(['GET'])
# @permission_classes([IsAuthenticated, ])
# def mylike(request):
#     user = request.user
#     storys = Story.objects.all()
#     my_like = []
#     for story in storys.values('id', 'like_users'):
#         if story['like_users'] == user.id:
#             my_like.append(story['id'])
#     print(my_like)


def test(request):
    # predict()
    return


@api_view(["POST"])
@permission_classes(
    [IsAuthenticated,]
)
def create_book(request):
    if request.method == "POST":
        book = get_object_or_404(Book, id=id)
        serializer = BookCreateSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            book = serializer.save(user=request.user)
            serializer = BookSerializer(book)
            return JsonResponse(serializer.data, safe=False)
        return HttpResponse(status=400)
    elif request.method == "DELETE":
        review = get_object_or_404(Book, id=id)
        review.delete()
        return
