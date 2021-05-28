[![Python application](https://github.com/Morbotu/drone-PWS/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Morbotu/drone-PWS/actions/workflows/python-app.yml)

# drone-PWS

Een automatische drone voor mijn profielwerkstuk 2021.

---

### Te doen

-   [ ] **Nvidia Jetson Nano computer vision**

    -   [x] Stel de Nvidia Jetson Nano in

        -   [x] Vind en download het OS
        -   [x] Installer het OS
        -   [x] Installeer camera

        -   Links

            -   [Getting Started with Jetson Nano 2GB Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#write)
            -   [Is there a way to generate real time depthmap from single camera video in python/opencv?](https://stackoverflow.com/questions/64685185/is-there-a-way-to-generate-real-time-depthmap-from-single-camera-video-in-python)

        -   Github

            -   [Morbotu/monodepth2](https://github.com/Morbotu/monodepth2)

            -   [tum-vision/lsd_slam](https://github.com/tum-vision/lsd_slam)

    -   [x] Maak programma dat diepte kan zien

        -   [x] Maak [live_camera_mac.py](live_camera_mac.py) schoon

    -   [ ] Schrijf programma voor camera

        -   [ ] Vind API voor image recognition
        -   [ ] Vind voorbeeld programma
        -   [ ] Schrijf eigen versie programma [0/4]

            -   [ ] Herken obstakels
            -   [ ] Stuur informatie door naar drone
            -   [ ] Herken object om te vervoeren
            -   [ ] Bereken beste route

        -   Links

            -   [Leveraging TensorFlow-TensorRT integration for Low latency Inference](https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html)
            -   [Create a real-time multiple object detection and recognition application in Python using with a Raspberry Pi Camera](https://maker.pro/nvidia-jetson/tutorial/deep-learning-with-jetson-nano-real-time-object-detection-and-recognition)

    -   [ ] Verbind Nvidia Jetson Nano met pixhawk 4

        -   [ ] Vind links op internet met voorbeelden
        -   [ ] Test verbinding uit en kijk of de drone bestuurd kan worden

        -   Links

            -   [How to Connect Jetson Nano to Pixhawk](https://forums.developer.nvidia.com/t/how-to-connect-jetson-nano-to-pixhawk/80189/3)

-   [ ] **Pixhawk 4 drone bouwen**

    -   [ ] Zet drone in elkaar

        -   [ ] Volg handleiding voor het in elkaar zetten van de drone
        -   [ ] Kijk of er ruimte is voor de Nvidia Jetson Nano
        -   [ ] Maak constructie voor grijparm

    -   [ ] Bestuur drone

        -   [ ] Vind programma's om de drone met de computer te besturen
        -   [ ] Test alle onderdelen uit
        -   [ ] Calibreer de drone
