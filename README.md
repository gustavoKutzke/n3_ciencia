# 🔍 Image Matcher ORB + RANSAC

Aplicação em **Python** que compara duas imagens do mesmo objeto em **ângulos diferentes**, detectando pontos de interesse e traçando linhas entre os pontos equivalentes.  
A interface gráfica é feita em **Tkinter** e o processamento de imagem em **OpenCV**.

---

## 🧠 O que o projeto faz

1. **Lê duas imagens do disco** (formatos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`).
2. **Redimensiona as duas imagens para a mesma altura**, mantendo a proporção (evita áreas pretas desnecessárias).
3. **Detecta pontos de interesse** nas duas imagens usando **ORB** (Oriented FAST and Rotated BRIEF).
4. **Compara descritores** com:
   - `BFMatcher` (Hamming)
   - Teste de razão de Lowe (ratio test)
5. **Filtra correspondências boas** e estima uma **homografia** usando **RANSAC**.
6. **Traça linhas entre pontos equivalentes** das duas imagens.
7. **Gera e salva resultados** na pasta `resultados/`:
   - `img1_keypoints.png` – pontos detectados na imagem 1  
   - `img2_keypoints.png` – pontos detectados na imagem 2  
   - `resultado_inliers_linhas.png` – linhas + pontos entre as imagens  
   - `resultado_linhas_sem_pontos_extras.png` – apenas as linhas  
   - `resultado_apenas_pontos.png` – canvas com os pontos inliers nas duas imagens

Tudo isso é feito de forma visual, com uma interface simples em Tkinter.

---

## 🛠 Tecnologias usadas

- [Python](https://www.python.org/) 3.x
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/) (para exibir imagens no Tkinter)
- Tkinter (GUI – já vem com o Python padrão)

---

## 📦 Requisitos

- Python 3.8+ (testado em 3.11)
- Pip instalado

### Dependências Python

Instale os pacotes com:

```bash
pip install opencv-python numpy Pillow
