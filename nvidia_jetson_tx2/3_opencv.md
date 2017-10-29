## 安裝 OpenCV

既然 TX2 上面有相機模組，那我們就來裝個 OpenCV 來做相機的影像處理吧！

Python3 會是我們的主要語言。

1. 安裝依賴套件

    ```sh
    $ sudo apt-get install build-essential cmake git pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev libatlas-base-dev gfortran
    ```

2. 取得 OpenCV 原始碼

    ```sh
    $ git clone git://github.com/Itseez/opencv
    $ cd opencv
    $ git checkout <你所要用的 OpenCV 版本，建議是用最新版>
    $ git clone git://github.com/Itseez/opencv_contrib
    $ cd opencv_contrib
    $ git checkout <你所要用的 OpenCV 版本，建議是用最新版>
    ```

3. 編譯 OpenCV

    ```sh
    $ mkdir opencv/build
    $ cd opencv/build
    $ cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib 所在的完整路徑> \
      -D BUILD_EXAMPLES=ON ..
    $ make -j4
    $ sudo make install
    $ sudo ldconfig
    ```

4. 安裝 OpenCV

    1. 建立虛擬開發環境

        ```sh
        $ virtualenv OpenCV
        ```

    2. 進入虛擬開發環境

        ```sh
        $ cd OpenCV
        $ source bin/active
        ```

    3. 找到 OpenCV

        ```sh
        $ ls -al /usr/local/lib/python<Python3 的版本號，如：3.5>/dist-packages/cv2*
        ```

    4. 連接 OpenCV 到虛擬環境

        ```sh
        $ ln -s <上一步驟印出的完整路徑> lib/python<Python3 的版本號，如：3.5>/site-packages/cv2.so
        ```

    5. 測試 OpenCV

        ```py
        import cv2
        cv2.__version__
        ```

        輸出

        ```
        '<OpenCV 的版本號>'
        ```
