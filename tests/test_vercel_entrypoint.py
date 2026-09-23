import importlib


def test_vercel_entrypoint_exports_flask_app():
    module = importlib.import_module('api.index')
    client = module.app.test_client()
    response = client.get('/')
    assert response.status_code == 200
