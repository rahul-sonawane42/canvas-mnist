import sys
import os
import pygame
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend")
sys.path.append(backend_dir)

from core.layers import Dense, Relu, Softmax


def load_model():
    print("Loading 3-layer weights into memory...")
    dense1 = Dense(784, 256)
    relu1 = Relu()
    dense2 = Dense(256, 128)
    relu2 = Relu()
    dense3 = Dense(128, 10)
    softmax = Softmax()

    models_dir = os.path.join(backend_dir, "models")
    dense1.weights = np.load(os.path.join(models_dir, "dense1_weights.npy"))
    dense1.biases = np.load(os.path.join(models_dir, "dense1_biases.npy"))
    dense2.weights = np.load(os.path.join(models_dir, "dense2_weights.npy"))
    dense2.biases = np.load(os.path.join(models_dir, "dense2_biases.npy"))
    dense3.weights = np.load(os.path.join(models_dir, "dense3_weights.npy"))
    dense3.biases = np.load(os.path.join(models_dir, "dense3_biases.npy"))

    return (dense1, relu1, dense2, relu2, dense3, softmax)


def preprocess(canvas_surface):
    raw = pygame.surfarray.array3d(canvas_surface)
    gray = cv2.cvtColor(np.transpose(raw, (1, 0, 2)), cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = thresh[y : y + h, x : x + w]

        if w > h:
            new_w = 20
            new_h = int(h * (20 / w))
        else:
            new_h = 20
            new_w = int(w * (20 / h))

        resized = cv2.resize(
            cropped, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA
        )

        canvas_28 = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - resized.shape[1]) // 2
        y_offset = (28 - resized.shape[0]) // 2
        canvas_28[
            y_offset : y_offset + resized.shape[0],
            x_offset : x_offset + resized.shape[1],
        ] = resized

        return canvas_28

    return cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)


def predict(model, small):
    dense1, relu1, dense2, relu2, dense3, softmax = model
    x = (small.astype(np.float32) / 255.0).reshape(1, 784)

    out1 = dense1.forward(x)
    act1 = relu1.forward(out1)
    out2 = dense2.forward(act1)
    act2 = relu2.forward(out2)
    out3 = dense3.forward(act2)
    return softmax.forward(out3)[0]


W, H = 960, 600

INK = (0, 0, 0)
PANEL = (12, 12, 12)
PANEL_HI = (30, 30, 30)
EDGE = (46, 46, 46)
CHALK = (240, 240, 240)
DIM = (140, 140, 140)
MINT = (94, 234, 212)
BAR = (72, 72, 72)
AMBER = (251, 191, 36)

CANVAS_SIZE = 400
CANVAS_POS = (52, 96)
BRUSH = 22


def font(size, bold=False):
    return pygame.font.SysFont("segoeui,helveticaneue,inter,arial", size, bold=bold)


def card(surface, rect, color=PANEL, outline=EDGE, radius=16):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if outline:
        pygame.draw.rect(surface, outline, rect, width=1, border_radius=radius)


def text(surface, f, s, color, pos, anchor="topleft"):
    img = f.render(s, True, color)
    r = img.get_rect(**{anchor: pos})
    surface.blit(img, r)
    return r


def draw_button(surface, f, rect, label, hovered, primary=False):
    if primary:
        bg = (130, 244, 226) if hovered else MINT
        fg = INK
        pygame.draw.rect(surface, bg, rect, border_radius=12)
    else:
        bg = PANEL_HI if hovered else PANEL
        fg = CHALK
        card(surface, rect, bg, EDGE, 12)
    text(surface, f, label, fg, rect.center, "center")


def main():
    model = load_model()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Canvas-to-Matrix | Neural Recognizer")

    f_title = font(26, True)
    f_sub = font(15)
    f_label = font(15, True)
    f_small = font(14)
    f_btn = font(16, True)
    f_digit = font(116, True)
    f_conf = font(22, True)

    canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    canvas.fill((0, 0, 0))
    canvas_rect = pygame.Rect(*CANVAS_POS, CANVAS_SIZE, CANVAS_SIZE)

    btn_clear = pygame.Rect(52, 524, 400, 44)

    drawing = False
    last = None
    has_ink = False
    dirty = False
    last_infer = 0

    probs = np.zeros(10)
    shown = np.zeros(10)
    preview = None
    pred_class = None

    def run_inference():
        nonlocal probs, preview, pred_class
        small = preprocess(canvas)
        probs = predict(model, small)
        pred_class = int(np.argmax(probs))
        rgb = np.stack([small.T] * 3, axis=-1).astype(np.float32) / 255.0
        rgb = (rgb * np.array(AMBER)).astype(np.uint8)
        preview = pygame.transform.scale(pygame.surfarray.make_surface(rgb), (112, 112))

    def clear():
        nonlocal has_ink, probs, preview, pred_class, last, dirty
        canvas.fill((0, 0, 0))
        has_ink = False
        dirty = False
        probs = np.zeros(10)
        preview = None
        pred_class = None
        last = None

    clock = pygame.time.Clock()

    while True:
        now = pygame.time.get_ticks()
        mouse = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if canvas_rect.collidepoint(event.pos):
                    drawing = True
                    p = (event.pos[0] - CANVAS_POS[0], event.pos[1] - CANVAS_POS[1])
                    pygame.draw.circle(canvas, (255, 255, 255), p, BRUSH)
                    last = p
                    has_ink = dirty = True
                elif btn_clear.collidepoint(event.pos):
                    clear()

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if drawing and has_ink:
                    run_inference()
                    dirty = False
                drawing = False
                last = None

            elif event.type == pygame.MOUSEMOTION and drawing:
                p = (event.pos[0] - CANVAS_POS[0], event.pos[1] - CANVAS_POS[1])
                p = (
                    max(0, min(CANVAS_SIZE, p[0])),
                    max(0, min(CANVAS_SIZE, p[1])),
                )
                if last is not None:
                    pygame.draw.line(canvas, (255, 255, 255), last, p, BRUSH * 2)
                pygame.draw.circle(canvas, (255, 255, 255), p, BRUSH)
                last = p
                dirty = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    clear()

        if drawing and dirty and has_ink and now - last_infer > 120:
            run_inference()
            last_infer = now
            dirty = False

        shown += (probs - shown) * 0.22

        screen.fill(INK)

        text(screen, f_title, "Canvas-to-Matrix", CHALK, (52, 30))
        text(
            screen,
            f_sub,
            "Draw a digit. A 784-128-10 network, written in NumPy, reads it live.",
            DIM,
            (52, 62),
        )

        card(screen, pygame.Rect(40, 84, 424, 424))
        screen.blit(canvas, CANVAS_POS)
        pygame.draw.rect(screen, EDGE, canvas_rect, width=1, border_radius=4)

        if not has_ink:
            text(
                screen,
                f_label,
                "Draw a digit from 0 to 9",
                DIM,
                canvas_rect.center,
                "center",
            )

        if canvas_rect.collidepoint(mouse):
            pygame.draw.circle(screen, (255, 255, 255), mouse, BRUSH, width=1)

        draw_button(
            screen, f_btn, btn_clear, "Clear  (C)", btn_clear.collidepoint(mouse)
        )

        card(screen, pygame.Rect(484, 84, 436, 176))
        if pred_class is not None:
            conf = probs[pred_class] * 100
            text(screen, f_digit, str(pred_class), MINT, (528, 172), "center")
            text(screen, f_conf, f"{conf:.1f}% confident", CHALK, (586, 112))
            verdict = (
                "Clear read."
                if conf >= 90
                else "Fairly sure."
                if conf >= 60
                else "Unsure. Try a bolder stroke."
            )
            text(screen, f_small, verdict, DIM, (586, 144))
        else:
            text(screen, f_digit, "?", EDGE, (528, 172), "center")
            text(screen, f_conf, "No drawing yet", CHALK, (586, 112))
            text(screen, f_small, "Draw on the canvas to start.", DIM, (586, 144))

        pv = pygame.Rect(790, 112, 112, 112)
        pygame.draw.rect(screen, (0, 0, 0), pv, border_radius=6)
        if preview:
            screen.blit(preview, pv)
        pygame.draw.rect(screen, EDGE, pv, width=1, border_radius=6)
        text(
            screen,
            f_small,
            "Model input, 28×28",
            DIM,
            (pv.centerx, pv.bottom + 14),
            "center",
        )

        card(screen, pygame.Rect(484, 276, 436, 292))
        text(screen, f_label, "Probability by digit", CHALK, (508, 292))

        for i in range(10):
            y = 326 + i * 24
            win = pred_class == i
            text(screen, f_small, str(i), CHALK if win else DIM, (508, y), "topleft")
            track = pygame.Rect(534, y + 3, 292, 12)
            pygame.draw.rect(screen, INK, track, border_radius=6)
            fill_w = int(track.width * float(shown[i]))
            if fill_w > 0:
                pygame.draw.rect(
                    screen,
                    MINT if win else BAR,
                    (track.x, track.y, max(fill_w, 8), track.height),
                    border_radius=6,
                )
            text(
                screen,
                f_small,
                f"{probs[i] * 100:5.1f}%",
                CHALK if win else DIM,
                (896, y),
                "topright",
            )

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
