import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

FACE_INFO = {
    1: {'name': 'Plano R-G (B constante)'},
    2: {'name': 'Plano R-G (B constante, oposto)'},
    3: {'name': 'Plano G-B (R constante)'},
    4: {'name': 'Plano G-B (R constante, oposto)'},
    5: {'name': 'Plano R-B (G constante)'},
    6: {'name': 'Plano R-B (G constante, oposto)'},
}

def generate_slice(face, slice_index, resolution=256):
    """
    Gera uma única imagem de fatia do cubo RGB.

    Args:
        face (int): O número da face de referência (1 a 6).
        slice_index (int): O índice da fatia (0 a 255).
        resolution (int): A resolução da imagem gerada.

    Returns:
        np.ndarray: Uma imagem (array NumPy) representando a fatia do cubo RGB.
    """
    slice_index = np.clip(slice_index, 0, 255)

    axis_range = np.linspace(0, 255, resolution, dtype=np.uint8)
    slice_img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    grid_y, grid_x = np.meshgrid(axis_range, axis_range)

    if face in [1, 2]:
        slice_value = slice_index if face == 1 else 255 - slice_index
        slice_img[:, :, 0] = grid_y  # Vermelho
        slice_img[:, :, 1] = grid_x  # Verde
        slice_img[:, :, 2] = slice_value  # Azul

    elif face in [3, 4]:
        slice_value = slice_index if face == 3 else 255 - slice_index
        slice_img[:, :, 0] = slice_value  # Vermelho
        slice_img[:, :, 1] = grid_y      # Verde
        slice_img[:, :, 2] = grid_x      # Azul

    elif face in [5, 6]:
        slice_value = slice_index if face == 5 else 255 - slice_index
        slice_img[:, :, 0] = grid_y      # Vermelho
        slice_img[:, :, 1] = slice_value  # Verde
        slice_img[:, :, 2] = grid_x      # Azul

    return slice_img


def save_slice(slice_img, file_path):
    """
    Salva uma imagem de fatia em um arquivo, convertendo de RGB para BGR.
    """
    cv2.imwrite(file_path, cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))

def criar_pasta_figuras():
    """Cria a pasta para salvar as figuras, se não existir."""
    pasta = "figuras_rgb"
    if not os.path.exists(pasta):
        os.makedirs(pasta)
        print(f"Pasta '{pasta}' criada para salvar as imagens.")
    return pasta


def main():
    """Função principal que executa o menu interativo."""
    print("=== GERADOR (FUNCIONAL) DE FATIAS DO CUBO RGB ===")

    pasta = criar_pasta_figuras()

    while True:
        print("\n-------------------- MENU --------------------")
        print("Qual face do cubo você deseja fatiar?")
        for face_num, info in FACE_INFO.items():
            print(f"  Face {face_num}: {info['name']}")

        print("\nDigite 'sair' a qualquer momento para terminar o programa.")

        try:
            input_face = input("Escolha o número da face (1-6): ")
            if input_face.lower() == 'sair':
                break

            face = int(input_face)
            if not 1 <= face <= 6:
                print("\n[ERRO] O número da face deve ser um inteiro entre 1 e 6. Tente novamente.")
                continue

        except ValueError:
            print("\n[ERRO] Entrada inválida. Por favor, digite um número inteiro. Tente novamente.")
            continue

        try:
            input_slice = input(f"Agora, digite o índice da fatia para a face {face} (0-255): ")
            if input_slice.lower() == 'sair':
                break

            slice_idx = int(input_slice)
            if not 0 <= slice_idx <= 255:
                print("\n[ERRO] O índice da fatia deve ser um inteiro entre 0 e 255. Tente novamente.")
                continue

        except ValueError:
            print("\n[ERRO] Entrada inválida. Por favor, digite um número inteiro. Tente novamente.")
            continue

        print(f"\nGerando fatia da face {face} no índice {slice_idx}...")
        fatia_gerada = generate_slice(face, slice_idx)

        nome_arquivo = f"face{face}_fatia{slice_idx:03d}.png"
        caminho_completo = os.path.join(pasta, nome_arquivo)

        save_slice(fatia_gerada, caminho_completo)
        print(f"✅ Sucesso! Imagem salva em: {caminho_completo}")

        plt.figure(figsize=(8, 8))
        plt.imshow(fatia_gerada)
        plt.title(f"Face {face}: {FACE_INFO[face]['name']}\nFatia no índice {slice_idx}", fontsize=14)
        plt.axis('off')

    print("\nPrograma terminado. Até logo!")

if __name__ == "__main__":
    main()
