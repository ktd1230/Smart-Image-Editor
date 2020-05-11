# AI REQURIED LIBRARY

+ cuda 10.1
+ cuda 설치는 notion에 정리해놨어요 
+ conda activate pytorch_env
+ pytorch version  >=1.4 확인
  + pip show torch
  + 만약 1.4이상이 아니라면 conda uninstall pytorch
+ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
+ pip 설치할 때는 꼭 가상환경에서 실행시켜야 합니다!!
  + pip install nltk
  + pip install pandas
  + 
  + pip install setuptools
  + pip install mxnet==1.6.0
  + pip install gluonnlp >=0.8.3
  + pip install sentencepiece
  + pip install transformers

### AI 기능 실행하려면 반드시 실행 시켜야 합니다!

+ **AI 기능 실행하지 않으려면** 
  
  + S02P23C104/Back/Django/main/view.py 에서
  
    + **from AI.prediction import predict** **(이부분 주석)**
  
      ```PYTHON
  from .models import Story
      from .serializers import *
  import json
      
  
      @api_view(['POST'])
  @permission_classes([IsAuthenticated, ])
      def image_upload(request):
    file_path = ''
        text = ['test1','test2']
    image = []
        file_names = []
    for _file in request.FILES.getlist('images[]'):
          request.FILES['images[]'] = _file
      file_path, file_name, path = uploaded(request.FILES['images[]'])
          image.append(path)
      file_names.append(file_name)
        # text = predict(file_names,MEDIA_ROOT)(이부분 주석)
  
        return JsonResponse({'result':'true', 'text':text, 'image':image})
      ```
  ```
  
  ```
  
+ conda 환경이라면 
  
  + conda actviate pytorch_env
  
+ cd S02P23C014/Back/
  
  + pip install .
  + 단 python 버전이 3.6 이상이여야 합니다 꼭!

## DJango

```bash
$ conda activate pytorch_env

$ pip install python
$ conda install pytorch torchvision

$ pip install djangorestframework-jwt
$ pip install djangorestframework
$ pip install django-cors-headers
$ pip install -U drf-yasg
```

```bash
# django 실행
$ python manage.py runserver  

# DB
$ python manage.py makemigratinos
$ python manage.py migrate
```








#  Model information

### Image Captioning model

+ **Without Attention**
  + config: config2020-04-22-00-48-42.json
  + encoder : encoder2020-04-22-00-48-42.pth
  + decoder : decoder2020-04-22-00-48-42.pth
  + loss : 2.1(CrossEntropy)
+ **With Attention** 
  + 

### KoGPT2 model



## AI SERVICE

+ predict 호출하기전에!!!
+ prediction.py
  + predict(images,IMAGE_DIRECTORY,AI_DIRECTORY_PATH,model_type)
    + param 
      + images : 이미지 배열
      + IMAGE_DIRECTORY: images가 저장된 디렉토리
      + AI_DIRECTORY_PATH: AI DIRECTORY path(절대경로)
      + model_type(string)
        + "life","story","news"
    + return
      + 1가지 이상의 생성된 문장 배열