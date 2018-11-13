FROM tensorflow/tensorflow:1.11.0-gpu-py3

RUN apt-get update && apt-get install -y \
	vim

RUN apt-get update && apt-get install -y \
	python3-tk \
	cmake \
	wget \
	git \
	yasm \
	libjpeg-dev \
	libtiff-dev \
	libjasper-dev \
	libpng-dev \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libv4l-dev \
	libatlas-base-dev \
	gfortran \
	libtbb2 \
	libtbb-dev \
	libpq-dev \
	libgtk2.0-dev \
	libsndfile1 \
	vim \
	&& apt-get -y clean all \
	&& rm -rf /var/lib/apt/lists/*


WORKDIR /

# Download opencv and opencv_contrib

RUN wget https://github.com/Itseez/opencv/archive/3.2.0.zip -O opencv.zip \
    && unzip opencv.zip \
    && wget https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip -O opencv_contrib.zip \
    && unzip opencv_contrib \
    && mkdir /opencv-3.2.0/cmake_binary

RUN cd /opencv-3.2.0/cmake_binary \
    && cmake -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-3.2.0/modules \
    -DBUILD_TIFF=ON \
    -DBUILD_opencv_java=OFF \
    -DWITH_CUDA=OFF \
    -DENABLE_AVX=ON \
    -DWITH_OPENGL=ON \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_opencv_python3=ON \
    -DCMAKE_INSTALL_PREFIX=$(python3.5 -c "import sys; print(sys.prefix)") \
    -DPYTHON_EXECUTABLE=$(which python3.5) \
    -DPYTHON_INCLUDE_DIR=$(python3.5 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3.5 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
    && make -j6 install \
    && rm /opencv.zip \
    && rm /opencv_contrib.zip \
    && rm -r /opencv-3.2.0 \
    && rm -r /opencv_contrib-3.2.0 \
    && cd /usr/lib/python3.5/dist-packages \
    && ln -s cv2.cpython-35m-x86_64-linux-gnu.so cv2.so

ARG KERAS_VERSION=2.2.4
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

#RUN pip3 install tqdm scikit-image pydicom xgboost
RUN pip3 install tqdm scikit-image pydicom xgboost Pillow imageio Shapely imgaug
RUN pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
RUN pip3 install fastai
RUN pip3 install alt-model-checkpoint

WORKDIR /handwritten-number-classification
