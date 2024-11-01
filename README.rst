Face Recognition System
=======================

| Recognize and manipulate faces from Python or from the command line
  with
| the world's simplest face recognition library.

| This also provides a simple ``face_recognition`` command line tool
  that lets
| you do face recognition on a folder of images from the command line!

| |PyPI|
| |Build Status|
| |Documentation Status|

Features
--------

Find faces in pictures
^^^^^^^^^^^^^^^^^^^^^^

Find all the faces that appear in a picture:

|image3|

.. code:: python

    import face_recognition
    image = face_recognition.load_image_file("your_file.jpg")
    face_locations = face_recognition.face_locations(image)

Find and manipulate facial features in pictures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the locations and outlines of each person's eyes, nose, mouth and
chin.

|image4|

.. code:: python

    import face_recognition
    image = face_recognition.load_image_file("your_file.jpg")
    face_landmarks_list = face_recognition.face_landmarks(image)

| Finding facial features is super useful for lots of important stuff.
  But you can also use for really stupid stuff
| like applying `digital
  make-up <https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py>`__
  (think 'Meitu'):

|image5|

Identify faces in pictures
^^^^^^^^^^^^^^^^^^^^^^^^^^

Recognize who appears in each photo.

|image6|

.. code:: python

    import face_recognition
    known_image = face_recognition.load_image_file("biden.jpg")
    unknown_image = face_recognition.load_image_file("unknown.jpg")

    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

You can even use this library with other Python libraries to do
real-time face recognition:

|image7|




.. |image3| image:: https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png
.. |image4| image:: https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png
.. |image5| image:: https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png
.. |image6| image:: https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png
.. |image7| image:: https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif


