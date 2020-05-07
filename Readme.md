## 정리된 아이디어

+ 집을 구하는 사용자를 위해 다각도의 시각에서 방을 구체적으로 살펴보게 해주는 서비스
  + 기술
    + multiple images -> 3d reconstruction ?
    + multiple images -> 2.5d reconstruction ?
+ 이동 주행 로봇 서비스를 통한 실내 공간 매핑 및 이를 이용한 청소?
  + 기술
    + ORB-SLAM 알고리즘을 이용
+ 작곡 서비스 및 악보 연주
  + 기술
    + GPT-2 모델을 이용
+ 집을 구하는 이를 위한 이미지 기반 방 추천 서비스방을 가상화 한 뒤 가구배치 등 인테리어
  + 기술
    + image retrieval
+ 손상된 이미지 복원 및 화질을 높여주는 서비스
  + 기술
    + GAN 등등...
+ Detection, Segmentation, Tracking, Classifying을 이용한 서비스
  + Classifying
    + 손동작으로 애플리케이션을 수행하는 서비스
      + 카카오, 유튜브 등을 실행시켜주는 서비스
      + 게임을 하는 서비스
  + Detection
    + No Idea
  + Segmentation
    + 이미지에서 사용자의 선택에 따라 특정 객체를 삭제하고 삭제한 부분을 주위의 배경으로 채워주는 서비스
  + Tracking
    + 이미지에서 사람의 동선을 Tracking하고 이를 기록하여 빈도에 따른 광고 공간 추천 서비스

## 개인적으로 하고 싶은 프로젝트

+ 경석:
  + 임베디드에서 영상 기반 알고리즘을 가속화 하기 위한 방법 탐색 및 수행
    + 새로운 BackBone 네트워크 구축
    + 모델 경량화
      + half precision
      + Quantization 등등..
  + 하드웨어 자원을 충분히 사용해서 가속화하는 방법 탐색 및 수행
    + Gpu를 이용한 가속화
      + ACL
      + OpenCL
    + Cpu를 이용한 가속화
      + Neon-API
      + Multi-threads
    + Cpu , Gpu 파이프라이닝
+ 창현:
  + RPA
  + MES(스마트 팩토리)
  + GAN Model
  + ERP, SAP