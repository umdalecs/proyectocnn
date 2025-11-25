import os
import shutil
import random
import math
from pathlib import Path

path_train = os.path.join(Path(__file__).parent, "train")
path_val = os.path.join(Path(__file__).parent, "val")
porcentaje = 0.25


def equilibrar_carpetas(train_dir, val_dir, ratio):
    subs_train = {
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    }

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        subs_val = set()
    else:
        subs_val = {
            d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))
        }

    carpetas_faltantes = subs_train - subs_val

    print(f"Carpetas encontradas en Train: {len(subs_train)}")
    print(f"Carpetas encontradas en Val: {len(subs_val)}")
    print(f"Carpetas a procesar (faltantes en Val): {len(carpetas_faltantes)}")
    print("-" * 30)

    for carpeta in carpetas_faltantes:
        ruta_origen = os.path.join(train_dir, carpeta)
        ruta_destino = os.path.join(val_dir, carpeta)

        os.makedirs(ruta_destino, exist_ok=True)

        archivos = [
            f
            for f in os.listdir(ruta_origen)
            if os.path.isfile(os.path.join(ruta_origen, f))
        ]

        total_archivos = len(archivos)
        cantidad_a_mover = math.ceil(total_archivos * ratio)  # Redondeamos hacia arriba

        if total_archivos > 0:
            archivos_seleccionados = random.sample(archivos, cantidad_a_mover)

            print(
                f"Procesando '{carpeta}': Moviendo {cantidad_a_mover} de {total_archivos} archivos."
            )

            for archivo in archivos_seleccionados:
                src = os.path.join(ruta_origen, archivo)
                dst = os.path.join(ruta_destino, archivo)
                shutil.move(src, dst)
        else:
            print(f"La carpeta '{carpeta}' está vacía, se ignoró.")

    print("-" * 30)
    print("Proceso completado.")


if __name__ == "__main__":
    if os.path.exists(path_train) and os.path.exists(path_val):
        equilibrar_carpetas(path_train, path_val, porcentaje)
    else:
        print(
            "Error: Revisa que las rutas de las carpetas 'train' y 'val' sean correctas."
        )
