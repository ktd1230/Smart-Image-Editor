## DJango 설정

1. visual code로 S02P31C101를 OPEN한다
2. ctrl + <shift> + p 를 눌러 명령 팔레트를  연다
3. 명령 팔레트에 Python:Select interpreter를 입력하여  AI를 작업했던 가상환경의 interpreter를 선택한다
   1. ex) conda의 가상환경 이름이 pytorch_env라면
   2. Anaconda3/envs/pytorch_env/python.exe를 interpreter로 설정한다
   3. Python:Select Interpreter가 나오지 않으면 검색을 해서.... 해결하도록
4. 설정이 끝났으면 visual code의 terminal을 git bash가 아닌 cmd로 설정한다
5. 아래에 따라 입력한다.

```bash

$ conda activate "가상환경이름"

$ pip install djangorestframework-jwt
$ pip install djangorestframework
$ pip install django-cors-headers
$ pip install -U drf-yasg
```

```bash
# DB
$ python manage.py makemigrations
$ python manage.py migrate

# django 실행
$ python manage.py runserver  
```








#  Vue 설정

1. 먼저 node js를 설치한 적이 없다면 node.js를 먼저 설치한다

2. 터미널에서 다음 명령어를 입력한다

   ```bash
   $ npm install -g @vue/cli
   $ cd S02P31C101/Front/
   $ npm install .
   $ npm run serve
   ```

## AI SERVICE

1. S02P31C101/Back/Django/main/views.py에 가면 다음 함수들이 있음.
   1. mask_rcnn
   2. resolution_up
   3. inpainting
2. 각자 해당하는 함수가 동작하는지 확인한다.
   1. 아쉽게도 동작을 확인하려면 Front,Django 전부다 실행을 시키고 웹에서 버튼을 눌러가며 테스트를 해봐야 한다.
3. Back에서 AI함수를 호출하는게 안될 것 
   1. pip install . 을 이용하는 방법이 있고
   2. 기타 등등이 있을 듯
   3. 특화 프로젝트 때는 pip install . 을 통해 AI 패키지 자체를 빌드해서 해결했다

# REQUIRED

+ mask-rcnn training
  + pycocotools
    + pip3 install numpy==1.17.4
    + pip3 install Cython
    + window :
      + pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
    + linux:
      + git clone https://github.com/cocodataset/cocoapi.git
      + cd PythonAPI
      + make

+ EDSR
  + numpy
  + scikit-image
  + imageio
  + matplotlib
  + tqdm
  + opencv-python
  + PIL

# ProSR 설정

환경은 반드시 Python 3.8 이전 버전 사용할 것

# Install torch
conda install pytorch=0.4.1 torchvision cuda91 -c pytorch

# Install image libraries
conda install scikit-image cython

# Install visdom
conda install visdom dominate -c conda-forge

# Install python dependencies
python3.7 -m pip install easydict pillow

Search Path
export PYTHONPATH=$PROJECT_ROOT/lib:$PYTHONPATH to include proSR into the search path.

Window일경우 제어판- 시스템- 환경변수에서 PYTHONPATH 이름 지정해주고 경로 설정 해주면 됨
