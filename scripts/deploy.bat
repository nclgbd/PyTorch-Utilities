bumpversion patch
scripts\build.bat
docker build -t nclgbd/pytorchdev:develop --no-cache .
docker run -it --rm nclgbd/pytorchdev:develop
