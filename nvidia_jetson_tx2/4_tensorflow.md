## 安裝 TensorFlow

1. 安裝依賴套件

    ```sh
    $ sudo apt-get install default-jdk libcupti-dev
    $ export JAVA_HOME='/usr/lib/jvm/java-8-openjdk-arm64/'
    ```

2. 取得 TensorFlow 編譯腳本

    ```sh
    $ git clone git://github.com/jetsonhacks/installTensorFlowTX2
    $ cd installTensorFlowTX2
    ```

3. 執行編譯腳本

    ```sh
    $ ./installPrerequisitesPy3.sh
    $ ./cloneTensorFlow.sh
    $ ./setTensorFlowEVPy3.sh
    $ ./buildTensorFlow.sh
    $ ./packageTensorFlow.sh
    ```

4. 安裝 TensorFlow

    1. 建立虛擬開發環境

        ```sh
        $ virtualenv TensorFlow
        ```

    2. 進入虛擬開發環境

        ```sh
        $ cd TensorFlow
        $ source bin/active
        ```

    3. 安裝 TensorFlow 到虛擬環境

        ```sh
        pip3 install $HOME/<TensorFlow 的 .whl 安裝封包>
        ```

    4. 測試 TensorFlow

        1. Hello World

            ```py
            import tensorflow as tf
            hello = tf.constant('Hello, TensorFlow on NVIDIA Jetson TX2!')
            sess = tf.Session()
            print(sess.run(hello))
            ```

            輸出

            ```
            Hello, TensorFlow on NVIDIA Jetson TX2!
            ```

        2. 運算單元

            ```py
            import tensorflow as tf
            # Creates a graph.
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)
            # Creates a session with log_device_placement set to True.
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            # Runs the op.
            print(sess.run(c))
            ```

            輸出

            ```
            name: NVIDIA Tegra X2
            major: 6 minor: 2 memoryClockRate (GHz) 1.3005
            MatMul: (MatMul): /job:localhost/replica:0/task:0/gpu:0
            b: (Const): /job:localhost/replica:0/task:0/gpu:0
            a: (Const): /job:localhost/replica:0/task:0/gpu:0
            [[ 22.  28.]
            [ 49.  64.]]
            ```
