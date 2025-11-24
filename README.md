## Crea el entorno virtual
```sh
python3 -m venv .venv
# Si estás en windows (PowerShell)
.venv\Scripts\Activate.ps1
# Si estás en MacOS o Linux (bash/zsh)
source .venv/bin/activate
```

## Instala las dependencias
```sh
pip install -r requirements.txt
```

## Para poblar el dataset 
```sh
python3 src/datapopulator.py --name "alejandro flores"
```

## Entrenar el modelo
```sh
python3 src/train.py
```

## Guardar dependencias
```sh
pip freeze > requirements.txt
```

## Ejecutar el proyecto
```sh
python3 src/main.py
```
