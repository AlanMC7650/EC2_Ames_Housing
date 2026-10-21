# Proyecto Integrador — Parte I (EC2) · Tasación de viviendas (Ames Housing)

SIS-351 Machine Learning · Metodología CRISP-DM

Modelo de regresión que estima el precio de venta de una vivienda en Ames (Iowa) a partir de sus
características, con baseline, modelos clásicos, aprendizaje no supervisado (PCA, K-Means, comparables)
y AutoML (FLAML) como benchmark.

## Estructura
| Carpeta | Contenido |
|---|---|
| `data/` | `AmesHousing.csv` (o instrucciones para obtenerlo en `data/README.md`) |
| `notebooks/` | `EC2_Ames_Housing_CRISP-DM.ipynb` — EDA y experimentos explicados |
| `src/` | `features.py` — ingeniería de features reutilizable (Parte II) |
| `models/` | Modelo elegido y artefactos no supervisados (`.joblib`), generados al ejecutar |
| `reports/` | Tablas comparativas (`.csv`) y figuras (`figures/`), generadas al ejecutar |
| `config/` | `experiment_config.json` y `automl_config.json`, generados al ejecutar |

## Cómo ejecutar
```bash
git clone https://github.com/AlanMC7650/EC2_Ames_Housing.git && cd EC2_Ames_Housing
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/EC2_Ames_Housing_CRISP-DM.ipynb   # Run All (~10 min en CPU)
```
Orden recomendado: ejecutar el notebook de arriba hacia abajo. La semilla (42) y todos los parámetros
están en la celda 0 y se guardan en `config/experiment_config.json`. Probado con Python 3.12 y las
versiones de `requirements.txt`.

`data/AmesHousing.csv` ya viene en el repositorio; si falta (p. ej. al abrir solo el notebook en
Google Colab), el notebook lo descarga automáticamente desde este repo. En Colab, ejecutar antes
`!pip install -q "flaml[automl]"` (Colab no lo trae instalado).

## Resultados (test, ejecución de referencia)
| Modelo | RMSE log | MAPE | R² |
|---|---|---|---|
| Baseline (calidad + tamaño) | 0.182 | 14.4 % | 0.837 |
| **Lasso (elegido)** | **0.108** | **7.8 %** | **0.947** |
| AutoML FLAML (LightGBM) | 0.110 | 7.4 % | 0.945 |

## Fuente del dataset
De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester
Regression Project.* Journal of Statistics Education, 19(3). Versión completa: 2 930 ventas, 2006–2010.
