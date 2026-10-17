# jarvis-hand-tracking

Controle por gestos com a câmera usando Python, OpenCV e MediaPipe.

- `jarvis.py`: interface estilo Jarvis. Faça a pinça (polegar + indicador) para criar retas no ar.
- `simulador.py`: simulador de física onde o dedo empurra bolinhas e a pinça as pega e arremessa.

## Como rodar

Use Python 3.12 (o MediaPipe pode não ter suporte para versões mais novas).

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
.venv\Scripts\python jarvis.py
```

Teclas do Jarvis: `Z` desfaz, `C` limpa, `F` tela cheia, `ESC` sai.
