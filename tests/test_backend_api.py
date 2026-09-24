"""
Sends photos to the Flask backend's /api/v1/analyze (in-process, no server
needed) and checks the response shape and error handling.
Uses one photo from ../face_shape_dataset; skips the real-photo check if it's missing.
Run: python tests/test_backend_api.py
"""
import io
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'webapp' / 'backend'))

from app.paths import DATASET_DIR
from app.face_shape_model import CLASS_NAMES
from backend_api import app

client = app.test_client()


def post(data):
    return client.post('/api/v1/analyze', data=data, content_type='multipart/form-data')


def test_missing_image_is_400():
    assert post({'gender': 'male'}).status_code == 400


def test_garbage_image_is_400():
    assert post({'image': (io.BytesIO(b'not an image'), 'x.jpg')}).status_code == 400


def test_real_photo():
    photos = sorted((DATASET_DIR / 'testing_set' / 'Oval').glob('*'))
    if not photos:
        print("  (skipped test_real_photo: dataset not found)")
        return
    r = post({'image': (open(photos[0], 'rb'), photos[0].name), 'gender': 'male'})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['faceShape'] in CLASS_NAMES
    assert body['glasses'] and body['hairstyle'] and body['outfitColors']['wear']
    ids = {g['id'] for g in body['allGlasses']}
    for rec in body['glasses']:
        assert set(rec['ids']) <= ids and rec['image'].endswith('.png'), rec
    print(f"  real photo -> {body['faceShape']} ({body['confidence']}), skin {body['skinTone']}, {len(ids)} frames")


def test_glasses_png_route():
    r = client.get('/glasses/glasses1.png')
    assert r.status_code == 200 and r.mimetype == 'image/png'
    assert r.data[1:4] == b'PNG'   # PNG files start with byte 0x89 then 'PNG' -- real image, not an error page
    assert client.get('/glasses/glasses13.png').status_code == 404   # hidden: brand logo
    assert client.get('/glasses/..%2Fmain.png').status_code == 404   # no path tricks


if __name__ == '__main__':
    test_missing_image_is_400()
    test_garbage_image_is_400()
    test_real_photo()
    test_glasses_png_route()
    print("test_backend_api: all passed")
