"""
WSGI config for sub3 project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

import sys
from pathlib import Path

base_path = Path(__file__).parent.absolute()
sys.path.append((base_path / "../").resolve().__str__())
sys.path.append((base_path / "../sub3").resolve().__str__())
sys.path.append((base_path / "../..").resolve().__str__())
sys.path.append((base_path / "../../AI").resolve().__str__())
sys.path.append((base_path / "../../AI/edsr_library").resolve().__str__())
sys.path.append((base_path / "../../AI/prosr_library").resolve().__str__())
sys.path.append((base_path / "../../AI/prosr_library/lib").resolve().__str__())
sys.path.append((base_path / "../../AI/inpainting_library").resolve().__str__())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sub3.settings')
application = get_wsgi_application()

# uwsgi_param QUERY_STRING $ query_string;
# uwsgi_param REQUEST_METHOD $ request_method;
# uwsgi_param CONTENT_TYPE $ content_type;
# uwsgi_param CONTENT_LENGTH $ content_length;

# uwsgi_param REQUEST_URI $ 요청 _uri;
# uwsgi_param PATH_INFO $ document_uri;
# uwsgi_param DOCUMENT_ROOT $ document_root;
# uwsgi_param SERVER_PROTOCOL $ server_protocol;
# uwsgi_param REQUEST_SCHEME $ scheme;
# uwsgi_param HTTPS $ https if_not_empty;

# uwsgi_param REMOTE_ADDR $ remote_addr;
# uwsgi_param REMOTE_PORT $ remote_port;
# uwsgi_param SERVER_PORT $ server_port;
# uwsgi_param SERVER_NAME $ server_name;
