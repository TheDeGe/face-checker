from flask import Flask, request
import numpy as np
import cv2
import io
import face_recognition
import hashlib

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload Photo</title>
    <h2>Upload Photo</h2>
    <form action="/upload_image" method=post enctype=multipart/form-data>
      <p><input type=file name=photo accept="image/*">
         <input type=submit>
    </form>
    '''
@app.route('/upload_image', methods=['POST'])
def upload_image():
    photo = request.files['photo']

    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    image_data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    print(face_encodings)

    top, right, bottom, left = face_locations[0]
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    if (img.shape[0] > 800) or (img.shape[1] > 1500):
        res_img = cv2.resize(img, (600, 800))
        cv2.imshow('photo', res_img)
    else:
        cv2.imshow('photo', img)
    cv2.waitKey()

    md5_hash = hashlib.md5()
    for encoding in face_encodings[0]:
        md5_hash.update(encoding)
    md5_hash = md5_hash.hexdigest()

    print(md5_hash)

    return photo.filename

if __name__ == '__main__':
    app.run(debug=True)