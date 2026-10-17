"""Simulador de movimento com a mão via câmera.

- A ponta do dedo indicador é o "sensor": ela empurra as bolinhas.
- Faça o gesto de pinça (polegar + indicador juntos) para PEGAR uma bolinha e arrastar.
- Teclas: R = reiniciar, ESC/Q = sair.
"""
import math
import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).with_name("hand_landmarker.task")
N_BOLAS = 12
GRAVIDADE = 900.0  # px/s²
QUICAR = 0.8
PINCA_LIMIAR = 0.35  # distância polegar-indicador / tamanho da mão
CONEXOES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Bola:
    def __init__(self, w, h):
        self.r = random.randint(18, 34)
        self.p = np.array([random.uniform(self.r, w - self.r),
                           random.uniform(self.r, h / 2)], dtype=float)
        self.v = np.array([random.uniform(-150, 150), 0.0])
        self.cor = tuple(int(c) for c in np.random.randint(80, 255, 3))
        self.presa = False


def criar_detector():
    opts = vision.HandLandmarkerOptions(
        # lê como bytes: o MediaPipe falha com acentos no caminho (ex.: "Área de Trabalho")
        base_options=BaseOptions(model_asset_buffer=MODEL.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(opts)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Não consegui abrir a câmera.")
    ok, frame = cap.read()
    h, w = frame.shape[:2]

    detector = criar_detector()
    bolas = [Bola(w, h) for _ in range(N_BOLAS)]
    ponta_ant = {}  # idx da mão -> posição anterior do dedo (para velocidade)
    segurando = {}  # idx da mão -> bola presa
    t_ant = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        agora = time.time()
        dt = min(agora - t_ant, 0.05)
        t_ant = agora

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect_for_video(img, int(agora * 1000))

        frame = (frame * 0.55).astype(np.uint8)  # escurece p/ destacar a simulação
        maos = []
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([[l.x * w, l.y * h] for l in lms])
            for a, b in CONEXOES:
                cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                         (255, 255, 255), 2)
            ponta = pts[8]
            tamanho = np.linalg.norm(pts[0] - pts[9]) + 1e-6
            pinca = np.linalg.norm(pts[4] - pts[8]) / tamanho < PINCA_LIMIAR
            vel = (ponta - ponta_ant.get(i, ponta)) / max(dt, 1e-3)
            ponta_ant[i] = ponta
            maos.append((i, ponta, vel, pinca, (pts[4] + pts[8]) / 2))
            cv2.circle(frame, tuple(ponta.astype(int)), 12,
                       (0, 255, 0) if pinca else (0, 200, 255), -1)

        ativas = {m[0] for m in maos}
        for i in list(segurando):
            if i not in ativas:
                segurando.pop(i).presa = False

        for i, ponta, vel, pinca, centro in maos:
            if pinca and i not in segurando:
                livres = [b for b in bolas if not b.presa
                          and np.linalg.norm(b.p - centro) < b.r + 30]
                if livres:
                    b = min(livres, key=lambda b: np.linalg.norm(b.p - centro))
                    b.presa = True
                    segurando[i] = b
            elif not pinca and i in segurando:
                b = segurando.pop(i)
                b.presa = False
                b.v = np.clip(vel, -2000, 2000)  # arremessa com a velocidade da mão
            if i in segurando:
                b = segurando[i]
                b.v = (centro - b.p) / max(dt, 1e-3)
                b.p = centro.copy()

        for b in bolas:
            if b.presa:
                continue
            b.v[1] += GRAVIDADE * dt
            b.v *= 0.999
            b.p += b.v * dt
            # empurrão pela ponta do dedo
            for _, ponta, vel, pinca, _ in maos:
                d = b.p - ponta
                dist = np.linalg.norm(d)
                if dist < b.r + 14 and dist > 1e-6:
                    n = d / dist
                    b.p = ponta + n * (b.r + 14)
                    b.v = b.v + n * max(0, -np.dot(b.v - vel, n)) * 1.0 + vel * 0.5
            # paredes
            if b.p[0] < b.r:
                b.p[0], b.v[0] = b.r, abs(b.v[0]) * QUICAR
            if b.p[0] > w - b.r:
                b.p[0], b.v[0] = w - b.r, -abs(b.v[0]) * QUICAR
            if b.p[1] < b.r:
                b.p[1], b.v[1] = b.r, abs(b.v[1]) * QUICAR
            if b.p[1] > h - b.r:
                b.p[1], b.v[1] = h - b.r, -abs(b.v[1]) * QUICAR
                b.v[0] *= 0.98

        # colisão bola x bola
        for a in range(len(bolas)):
            for c in range(a + 1, len(bolas)):
                A, B = bolas[a], bolas[c]
                d = B.p - A.p
                dist = np.linalg.norm(d)
                minimo = A.r + B.r
                if 1e-6 < dist < minimo:
                    n = d / dist
                    sobra = (minimo - dist) / 2
                    if not A.presa:
                        A.p -= n * sobra
                    if not B.presa:
                        B.p += n * sobra
                    rel = np.dot(A.v - B.v, n)
                    if rel > 0:
                        A.v -= n * rel
                        B.v += n * rel

        for b in bolas:
            cv2.circle(frame, tuple(b.p.astype(int)), b.r, b.cor, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(b.p.astype(int)), b.r, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "Dedo = empurra | Pinca = pega e arremessa | R = reiniciar | ESC = sair",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Simulador de movimento", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("r"):
            bolas = [Bola(w, h) for _ in range(N_BOLAS)]
            segurando.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
