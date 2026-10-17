"""Interface estilo Jarvis: crie retas no ar com o dedo.

- Faça a PINÇA (polegar + indicador) para fixar o ponto inicial.
- Mantendo a pinça, mexa a mão: uma reta holográfica acompanha o dedo.
- Solte a pinça para fixar a reta. Comece a próxima a partir de qualquer lugar.
- Pinçar perto da ponta de uma reta existente continua a partir dela (encadeia).
- Teclas: Z = desfazer, C = limpar, ESC/Q = sair.
"""
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).with_name("hand_landmarker.task")
PINCA_ON, PINCA_OFF = 0.30, 0.42  # histerese p/ não piscar
SNAP = 30  # px: raio para encaixar em pontas existentes
LARGURA, ALTURA = 1280, 720  # resolução pedida à câmera (usa a maior suportada)
CIANO = (255, 220, 40)  # BGR
LARANJA = (60, 170, 255)


def brilho(layer, p1, p2, cor, esp=2):
    """Desenha linha com efeito neon (camadas largas + núcleo branco)."""
    p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
    cv2.line(layer, p1, p2, cor, esp + 10, cv2.LINE_AA)
    cv2.line(layer, p1, p2, cor, esp + 4, cv2.LINE_AA)
    return p1, p2


def nucleo(frame, p1, p2, esp=2):
    cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (255, 255, 255), esp, cv2.LINE_AA)


def no(frame, p, cor, r=7):
    p = tuple(map(int, p))
    cv2.circle(frame, p, r + 5, cor, 1, cv2.LINE_AA)
    cv2.circle(frame, p, r - 2, (255, 255, 255), -1, cv2.LINE_AA)


def rotulo(frame, a, b, cor):
    m = ((a + b) / 2).astype(int)
    comp = np.linalg.norm(b - a)
    ang = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))
    cv2.putText(frame, f"{comp:.0f}px  {ang:.0f}deg", (m[0] + 8, m[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)


def mira(frame, p, cor, pinca):
    x, y = map(int, p)
    r = 16 if pinca else 26
    cv2.circle(frame, (x, y), r, cor, 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(frame, (x + dx * (r + 4), y + dy * (r + 4)),
                 (x + dx * (r + 14), y + dy * (r + 14)), cor, 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Não consegui abrir a câmera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    cv2.namedWindow("JARVIS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("JARVIS", w, h)
    tela_cheia = False

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL.read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(opts)

    retas = []  # lista de (p1, p2) fixadas
    inicio = None  # ponto inicial da reta em andamento
    pinçando = False
    suave = None  # posição suavizada do dedo
    grade_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(time.time() * 1000))

        # fundo escuro azulado + grade HUD
        frame = (frame * 0.35).astype(np.uint8)
        frame[..., 0] = np.clip(frame[..., 0].astype(int) + 25, 0, 255).astype(np.uint8)
        for x in range(0, w, 60):
            cv2.line(frame, (x, 0), (x, h), (60, 40, 15), 1)
        for y in range(0, h, 60):
            cv2.line(frame, (0, y), (w, y), (60, 40, 15), 1)

        dedo = None
        if res.hand_landmarks:
            pts = np.array([[l.x * w, l.y * h] for l in res.hand_landmarks[0]])
            bruto = pts[8]
            suave = bruto if suave is None else suave * 0.6 + bruto * 0.4
            dedo = suave
            tam = np.linalg.norm(pts[0] - pts[9]) + 1e-6
            razao = np.linalg.norm(pts[4] - pts[8]) / tam
            if not pinçando and razao < PINCA_ON:
                pinçando = True
                pontas = [p for r in retas for p in r]
                alvo = min(pontas, key=lambda p: np.linalg.norm(p - dedo), default=None)
                inicio = alvo.copy() if alvo is not None and np.linalg.norm(alvo - dedo) < SNAP \
                    else dedo.copy()
            elif pinçando and razao > PINCA_OFF:
                pinçando = False
                if inicio is not None and np.linalg.norm(dedo - inicio) > 8:
                    retas.append((inicio.copy(), dedo.copy()))
                inicio = None
            # mão em ciano
            for i in (0, 4, 8, 12, 16, 20):
                cv2.circle(frame, tuple(pts[i].astype(int)), 3, (160, 120, 30), -1, cv2.LINE_AA)
        else:
            suave = None
            if pinçando:  # perdeu a mão: cancela a reta em andamento
                pinçando, inicio = False, None

        # camada de brilho
        layer = np.zeros_like(frame)
        for a, b in retas:
            brilho(layer, a, b, CIANO)
        if pinçando and inicio is not None and dedo is not None:
            brilho(layer, inicio, dedo, LARANJA)
        layer = cv2.GaussianBlur(layer, (0, 0), 9)
        frame = cv2.add(frame, layer)

        for a, b in retas:
            nucleo(frame, a, b)
            no(frame, a, CIANO)
            no(frame, b, CIANO)
            rotulo(frame, a, b, CIANO)
        if pinçando and inicio is not None and dedo is not None:
            nucleo(frame, inicio, dedo)
            no(frame, inicio, LARANJA)
            rotulo(frame, inicio, dedo, LARANJA)
        if dedo is not None:
            mira(frame, dedo, LARANJA if pinçando else CIANO, pinçando)

        status = "TRACANDO" if pinçando else ("MAO DETECTADA" if dedo is not None else "AGUARDANDO MAO")
        cv2.putText(frame, f"J.A.R.V.I.S  |  {status}  |  RETAS: {len(retas)}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CIANO, 1, cv2.LINE_AA)
        cv2.putText(frame, "Pinca = inicia/solta reta  |  Z desfaz  |  C limpa  |  F tela cheia  |  ESC sai",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 160, 60), 1, cv2.LINE_AA)
        cv2.imshow("JARVIS", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("z") and retas:
            retas.pop()
        if k == ord("f"):
            tela_cheia = not tela_cheia
            cv2.setWindowProperty("JARVIS", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if tela_cheia else cv2.WINDOW_NORMAL)
        if k == ord("c"):
            retas.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
