import AI
from setuptools import find_packages, setup
setup(name='malaton',
      version=AI.__version__,      
      license='midified MIT',
      author='Kyung Seok',
      author_email='leek018@naver.com',
      description='Image-to-gpt',
      packages=find_packages(where=".", exclude=(
          'Django'          
      )),      
      zip_safe=False,
      include_package_data=True,
      )