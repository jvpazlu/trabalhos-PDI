"""
Implementação do filtro High-Boost para aumento de nitidez em imagens.

Este script aplica a técnica de filtragem high-boost em uma imagem de entrada
e salva o resultado automaticamente em uma pasta chamada 'resultados_highboost'.
O nome do arquivo de saída é gerado a partir do nome de entrada e do fator k.
"""

import cv2
import numpy as np
import argparse
import os

def high_boost_filter(image, k, kernel_size=(5, 5)):
    """
    Aplica o filtro High-Boost a uma imagem.
    (Esta função permanece exatamente a mesma)
    """
    image_float = image.astype(np.float64)
    blurred_image = cv2.GaussianBlur(image_float, kernel_size, 0)
    mask = image_float - blurred_image
    sharpened_image_float = image_float + k * mask
    sharpened_image_clipped = np.clip(sharpened_image_float, 0, 255)
    final_image = sharpened_image_clipped.astype(np.uint8)
    return final_image

def main():
    """
    Função principal para carregar a imagem, aplicar o filtro e salvar o resultado.
    """
    parser = argparse.ArgumentParser(description="Aplica filtragem High-Boost a uma imagem.")
    parser.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada.")
    parser.add_argument("-k", "--k_factor", type=float, required=True, help="Fator de boost k.")

    args = vars(parser.parse_args())

    output_folder_name = "resultados_highboost"
    if not os.path.exists(output_folder_name):
        print(f"Criando a pasta de resultados: '{output_folder_name}/'")
        os.makedirs(output_folder_name)

    input_image_path = args["image"]
    input_image = cv2.imread(input_image_path, 0)
    if input_image is None:
        print(f"Erro: Não foi possível carregar a imagem do caminho: {input_image_path}")
        return

    k_value = args["k_factor"]
    print(f"Aplicando filtro High-Boost com k = {k_value} em '{input_image_path}'...")
    sharpened_image = high_boost_filter(input_image, k=k_value)

    base_name = os.path.basename(input_image_path)
    name_without_ext, _ = os.path.splitext(base_name)
    k_str = str(k_value).replace('.', 'p')
    output_filename = f"{name_without_ext}_k{k_str}.png"

    full_output_path = os.path.join(output_folder_name, output_filename)

    cv2.imwrite(full_output_path, sharpened_image)
    print(f"✅ Sucesso! Imagem com nitidez realçada salva em: {full_output_path}")


if __name__ == "__main__":
    main()
