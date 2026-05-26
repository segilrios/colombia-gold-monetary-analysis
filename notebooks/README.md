# Notebooks para Kaggle

## Orden recomendado

1. `01_umap_local_biplot_kaggle.ipynb`
2. `02_random_forest_grey_box_kaggle.ipynb`

## Uso

Subir el repositorio como dataset de Kaggle o clonarlo dentro del notebook. Los notebooks buscan automaticamente `configs/project.yaml`, agregan `src/` al `PYTHONPATH` y escriben resultados en `/kaggle/working` cuando estan en Kaggle.

## Salidas principales

- `reports/figures/policy_umap_local_biplot_clusters.png`
- `reports/figures/policy_umap_by_executive_period.png`
- `reports/tables/random_forest_grey_box_metrics.csv`
- `reports/figures/random_forest_grey_box_predictions_h1.png`
- `reports/figures/random_forest_feature_importance_h1.png`
