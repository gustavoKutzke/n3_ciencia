import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


# ==============================
#   PARÂMETROS GERAIS
# ==============================
RATIO_TEST = 0.75              # teste de razão mais rígido
MIN_MATCHES_RANSAC = 25        # exige mais matches pra confiar na homografia
RANSAC_REPROJ_THRESH = 3.0     # tolerância menor no RANSAC
MAX_MATCHES_FOR_H = 200        # máximo de matches usados pra estimar H
MAX_LINES_TO_DRAW = 80         # máximo de linhas desenhadas na imagem final


# ==============================
#   FUNÇÕES DE PROCESSAMENTO
# ==============================

def redimensionar_para_altura(img, target_h):
    """
    Redimensiona uma imagem para ter altura = target_h,
    preservando a proporção.
    """
    h, w = img.shape[:2]
    if h == target_h:
        return img

    escala = target_h / h
    novo_w = int(w * escala)

    interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (novo_w, target_h), interpolation=interp)


def carregar_imagens(caminho_a, caminho_b):
    """
    1) Ler duas imagens do disco (coloridas)
    2) Redimensionar as duas para a MESMA ALTURA
    3) Gerar versões em tons de cinza a partir delas
    """
    try:
        img_a_color = cv2.imread(caminho_a, cv2.IMREAD_COLOR)
        img_b_color = cv2.imread(caminho_b, cv2.IMREAD_COLOR)

        if img_a_color is None or img_b_color is None:
            raise IOError("Não foi possível carregar uma ou ambas as imagens.")

        # mesma altura para minimizar área preta no drawMatches
        hA, wA = img_a_color.shape[:2]
        hB, wB = img_b_color.shape[:2]
        altura_alvo = min(hA, hB)

        img_a_color = redimensionar_para_altura(img_a_color, altura_alvo)
        img_b_color = redimensionar_para_altura(img_b_color, altura_alvo)

        # versões em cinza
        img_a_gray = cv2.cvtColor(img_a_color, cv2.COLOR_BGR2GRAY)
        img_b_gray = cv2.cvtColor(img_b_color, cv2.COLOR_BGR2GRAY)

        return img_a_gray, img_b_gray, img_a_color, img_b_color
    except Exception as e:
        messagebox.showerror("Erro ao carregar imagens", str(e))
        return None, None, None, None


def extrair_features_orb(img_gray):
    """Extrai keypoints e descritores usando ORB."""
    orb = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31
    )
    kp, desc = orb.detectAndCompute(img_gray, None)
    return kp, desc


def desenhar_keypoints(img_color, keypoints):
    """Destaca visualmente os pontos de interesse em uma única imagem."""
    return cv2.drawKeypoints(
        img_color, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def filtrar_correspondencias(desc_a, desc_b, max_matches=MAX_MATCHES_FOR_H):
    """
    Matching entre descritores com teste de razão de Lowe,
    retornando apenas os melhores matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pares_knn = bf.knnMatch(desc_a, desc_b, k=2)

    bons_matches = []
    for par in pares_knn:
        if len(par) == 2:
            m, n = par
            if m.distance < RATIO_TEST * n.distance:
                bons_matches.append(m)

    # Ordena pelos melhores (menor distance) e limita quantidade
    bons_matches = sorted(bons_matches, key=lambda m: m.distance)
    if len(bons_matches) > max_matches:
        bons_matches = bons_matches[:max_matches]

    return bons_matches


def estimar_homografia(kp_a, kp_b, matches):
    """
    Estima homografia via RANSAC e devolve H + máscara de inliers.
    """
    if len(matches) < MIN_MATCHES_RANSAC:
        return None, None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    return H, mask


def montar_canvas_pontos(img_a_color, img_b_color, kp_a, kp_b, matches_inliers):
    """
    Cria uma imagem lado a lado só com os keypoints inliers desenhados.
    """
    inliers_kp_a = [kp_a[m.queryIdx] for m in matches_inliers]
    inliers_kp_b = [kp_b[m.trainIdx] for m in matches_inliers]

    img_a_pts = cv2.drawKeypoints(
        img_a_color, inliers_kp_a, None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_b_pts = cv2.drawKeypoints(
        img_b_color, inliers_kp_b, None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    hA, wA = img_a_pts.shape[:2]
    hB, wB = img_b_pts.shape[:2]
    altura = max(hA, hB)
    largura = wA + wB

    canvas = np.zeros((altura, largura, 3), dtype=np.uint8)
    canvas[:hA, :wA] = img_a_pts
    canvas[:hB, wA:wA + wB] = img_b_pts

    return canvas


def salvar_resultados(img_linhas, img_linhas_simples, img_pontos,
                      img_kp_a=None, img_kp_b=None):
    """
    Salva todos os resultados em disco:
    - img_kp_a / img_kp_b: keypoints destacados em cada imagem
    - img_linhas: linhas + pontos entre as duas imagens
    - img_linhas_simples: só linhas
    - img_pontos: canvas de pontos inliers
    """
    try:
        pasta = "resultados"
        os.makedirs(pasta, exist_ok=True)

        if img_kp_a is not None:
            cv2.imwrite(os.path.join(pasta, "img1_keypoints.png"), img_kp_a)
        if img_kp_b is not None:
            cv2.imwrite(os.path.join(pasta, "img2_keypoints.png"), img_kp_b)

        cv2.imwrite(os.path.join(pasta, "resultado_inliers_linhas.png"), img_linhas)
        cv2.imwrite(os.path.join(pasta, "resultado_linhas_sem_pontos_extras.png"), img_linhas_simples)
        cv2.imwrite(os.path.join(pasta, "resultado_apenas_pontos.png"), img_pontos)
    except Exception as e:
        print("Erro ao salvar resultados:", e)


# ==============================
#   PIPELINE PRINCIPAL
# ==============================

def processar_comparacao(caminho_a, caminho_b):
    """
    Pipeline completo:
    - Lê as imagens
    - Redimensiona para mesma altura
    - Detecta & destaca pontos de interesse
    - Traça linhas entre pontos equivalentes
    - Salva todos os resultados
    """

    img_a_gray, img_b_gray, img_a_color, img_b_color = carregar_imagens(caminho_a, caminho_b)
    if img_a_gray is None:
        return None

    # 1) Detecta features nas duas imagens
    kp_a, desc_a = extrair_features_orb(img_a_gray)
    kp_b, desc_b = extrair_features_orb(img_b_gray)

    # Imagens com keypoints destacados individualmente
    img_a_kp = desenhar_keypoints(img_a_color, kp_a)
    img_b_kp = desenhar_keypoints(img_b_color, kp_b)

    # Se não achou descritores suficientes, só mostra os pontos
    if desc_a is None or desc_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        salvar_resultados(
            img_linhas=img_a_kp,
            img_linhas_simples=img_b_kp,
            img_pontos=montar_canvas_pontos(img_a_color, img_b_color, kp_a, kp_b, []),
            img_kp_a=img_a_kp,
            img_kp_b=img_b_kp
        )
        return img_a_kp

    # 2) Matching entre descritores (já limitado e ordenado)
    good_matches = filtrar_correspondencias(desc_a, desc_b)

    if len(good_matches) < MIN_MATCHES_RANSAC:
        img_matches_fracas = cv2.drawMatches(
            img_a_color, kp_a,
            img_b_color, kp_b,
            good_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        canvas_pontos = montar_canvas_pontos(img_a_color, img_b_color, kp_a, kp_b, good_matches)
        salvar_resultados(img_matches_fracas, img_matches_fracas, canvas_pontos, img_a_kp, img_b_kp)
        return img_matches_fracas

    # 3) Estima homografia e inliers
    H, mask = estimar_homografia(kp_a, kp_b, good_matches)

    if H is None or mask is None:
        img_sem_h = cv2.drawMatches(
            img_a_color, kp_a,
            img_b_color, kp_b,
            good_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        canvas_pontos = montar_canvas_pontos(img_a_color, img_b_color, kp_a, kp_b, good_matches)
        salvar_resultados(img_sem_h, img_sem_h, canvas_pontos, img_a_kp, img_b_kp)
        return img_sem_h

    mask = mask.ravel().astype(bool)
    matches_inliers = [m for m, inlier in zip(good_matches, mask) if inlier]

    if len(matches_inliers) < MIN_MATCHES_RANSAC:
        img_poucos_inliers = cv2.drawMatches(
            img_a_color, kp_a,
            img_b_color, kp_b,
            good_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        canvas_pontos = montar_canvas_pontos(img_a_color, img_b_color, kp_a, kp_b, good_matches)
        salvar_resultados(img_poucos_inliers, img_poucos_inliers, canvas_pontos, img_a_kp, img_b_kp)
        return img_poucos_inliers

    # --- NOVO: ordenar inliers por distância e limitar quantos vamos desenhar ---
    matches_inliers = sorted(matches_inliers, key=lambda m: m.distance)
    if len(matches_inliers) > MAX_LINES_TO_DRAW:
        matches_to_draw = matches_inliers[:MAX_LINES_TO_DRAW]
    else:
        matches_to_draw = matches_inliers

    # 4) Visualizações principais (com inliers filtrados)

    # 4.1) Linhas + pontos (inliers apenas)
    img_linhas = cv2.drawMatches(
        img_a_color, kp_a,
        img_b_color, kp_b,
        matches_to_draw, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    # 4.2) Apenas linhas
    img_linhas_simples = cv2.drawMatches(
        img_a_color, kp_a,
        img_b_color, kp_b,
        matches_to_draw, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 4.3) Canvas de pontos inliers (usando todos os inliers, não só os desenhados)
    img_pontos = montar_canvas_pontos(img_a_color, img_b_color, kp_a, kp_b, matches_inliers)

    # Salva tudo
    salvar_resultados(img_linhas, img_linhas_simples, img_pontos, img_a_kp, img_b_kp)

    # imagem exibida na interface
    return img_linhas


# ==============================
#   INTERFACE GRÁFICA (TKINTER)
# ==============================

img_a = ""
img_b = ""


def escolher_img_a():
    global img_a
    caminho = filedialog.askopenfilename(
        title="Escolha a primeira imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if caminho:
        img_a = caminho
        lbl_a.config(text=os.path.basename(caminho), fg="white")


def escolher_img_b():
    global img_b
    caminho = filedialog.askopenfilename(
        title="Escolha a segunda imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if caminho:
        img_b = caminho
        lbl_b.config(text=os.path.basename(caminho), fg="white")


def executar():
    if not img_a or not img_b:
        messagebox.showinfo("Atenção", "Selecione as duas imagens antes de comparar.")
        return

    lbl_status.config(text="Processando correspondências...", fg="#3ba55d")
    root.update_idletasks()

    resultado = processar_comparacao(img_a, img_b)

    if resultado is None:
        lbl_status.config(text="Falha no processamento.", fg="red")
        return

    lbl_status.config(text="Concluído! Resultados salvos na pasta 'resultados'.", fg="#3ba55d")

    # Redimensionar a imagem de saída para caber na janela
    largura_max = 1200
    altura_max = 700
    h, w = resultado.shape[:2]

    escala_larg = largura_max / w
    escala_alt = altura_max / h
    escala = min(escala_larg, escala_alt, 1.0)

    nova_w = int(w * escala)
    nova_h = int(h * escala)

    if escala < 1.0:
        resultado = cv2.resize(resultado, (nova_w, nova_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tk_img = ImageTk.PhotoImage(pil_img)

    lbl_out.config(image=tk_img)
    lbl_out.image = tk_img


# ============ CONFIG. JANELA ============

root = tk.Tk()
root.title("Comparador de Imagens - ORB + RANSAC (Lógica Alternativa)")
root.geometry("1280x800")
root.configure(bg="#1e1e1e")

COR_BTN = "#3ba55d"
COR_BTN_HOVER = "#49c46d"
COR_BG = "#1e1e1e"
COR_CARD = "#2b2b2b"


def on_enter(e):
    e.widget["background"] = COR_BTN_HOVER


def on_leave(e):
    e.widget["background"] = COR_BTN


# Topo: seleção de imagens
frame_top = tk.Frame(root, bg=COR_BG, pady=10)
frame_top.pack(fill=tk.X)

btn1 = tk.Button(
    frame_top, text="Selecionar Imagem 1", width=20,
    command=escolher_img_a, bg=COR_BTN, fg="white",
    activebackground=COR_BTN_HOVER, relief=tk.FLAT
)
btn1.pack(side=tk.LEFT, padx=10)
btn1.bind("<Enter>", on_enter)
btn1.bind("<Leave>", on_leave)

lbl_a = tk.Label(frame_top, text="Nenhuma imagem escolhida", fg="gray", bg=COR_BG, width=30, anchor="w")
lbl_a.pack(side=tk.LEFT, padx=(0, 20))

btn2 = tk.Button(
    frame_top, text="Selecionar Imagem 2", width=20,
    command=escolher_img_b, bg=COR_BTN, fg="white",
    activebackground=COR_BTN_HOVER, relief=tk.FLAT
)
btn2.pack(side=tk.LEFT, padx=10)
btn2.bind("<Enter>", on_enter)
btn2.bind("<Leave>", on_leave)

lbl_b = tk.Label(frame_top, text="Nenhuma imagem escolhida", fg="gray", bg=COR_BG, width=30, anchor="w")
lbl_b.pack(side=tk.LEFT)

# Botão principal
frame_mid = tk.Frame(root, bg=COR_BG, pady=10)
frame_mid.pack()

btn_compare = tk.Button(
    frame_mid,
    text="COMPARAR IMAGENS",
    width=30,
    height=2,
    bg=COR_BTN,
    fg="white",
    font=("Segoe UI", 12, "bold"),
    command=executar,
    activebackground=COR_BTN_HOVER,
    relief=tk.FLAT
)
btn_compare.pack()
btn_compare.bind("<Enter>", on_enter)
btn_compare.bind("<Leave>", on_leave)

lbl_status = tk.Label(
    root, text="Selecione duas imagens do mesmo objeto em ângulos diferentes.",
    font=("Segoe UI", 10, "italic"),
    fg="white",
    bg=COR_BG
)
lbl_status.pack(pady=5)

# Área da imagem de saída
frame_img = tk.Frame(root, bg=COR_CARD, borderwidth=1, relief=tk.SUNKEN)
frame_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

lbl_out = tk.Label(frame_img, bg=COR_CARD)
lbl_out.pack(fill=tk.BOTH, expand=True)

root.mainloop()
