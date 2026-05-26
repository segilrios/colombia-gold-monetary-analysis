# Implementacion Random Forest y caja gris

## Objetivo

Esta rama implementa modelos supervisados para pronosticar retornos futuros del oro usando la literatura de Random Forest, macroeconomic random forests y modelos de caja gris.

## Modelos incluidos

- `zero_return_baseline`: benchmark ingenuo de retorno futuro igual a cero.
- `decision_tree`: arbol interpretable de baja profundidad.
- `random_forest`: ensamble no lineal para capturar interacciones macrofinancieras.
- `grey_box_random_forest`: bloque macro lineal con RidgeCV mas Random Forest sobre residuos.

## Horizonte y frecuencia

La configuracion usa frecuencia mensual y horizontes de 1, 3 y 6 meses. La eleccion mensual es deliberada porque el analisis busca relacion con politica monetaria y periodos de gobierno.

## Uso local

```bash
pip install -r requirements.txt
python scripts/run_policy_umap_biplot.py --config configs/project.yaml
python scripts/run_random_forest_grey_box.py --config configs/project.yaml
```

El primer comando genera el contexto UMAP. El segundo entrena los modelos supervisados y, si existe `data/processed/policy_umap_embedding.csv`, usa el cluster UMAP como variable de regimen.

## Salidas

- `reports/tables/random_forest_grey_box_metrics.csv`
- `reports/tables/random_forest_grey_box_predictions_h1.csv`
- `reports/tables/random_forest_grey_box_predictions_h3.csv`
- `reports/tables/random_forest_grey_box_predictions_h6.csv`
- `reports/tables/random_forest_feature_importance_h1.csv`
- `reports/tables/random_forest_permutation_importance_h1.csv`
- `reports/tables/grey_box_residual_importance_h1.csv`
- `reports/figures/random_forest_grey_box_predictions_h1.png`
- `reports/figures/random_forest_feature_importance_h1.png`
- `reports/figures/grey_box_residual_importance_h1.png`

Las salidas equivalentes se generan para los horizontes 3 y 6.

## Primera corrida local

La primera corrida entrenada con frecuencia mensual produjo:

- Horizonte 1 mes: mejor RMSE con `random_forest`.
- Horizonte 3 meses: mejor RMSE con `decision_tree`.
- Horizonte 6 meses: mejor RMSE con `random_forest`.
- La caja gris mejoro despues de restringir el bloque lineal a cambios y variaciones, pero todavia no supera al Random Forest puro.

Lectura inicial:

- Las variables mas importantes para horizonte 1 incluyen cambios en Brent, memoria reciente del retorno del oro, diferencia del oro, TRM rezagada y DTF.
- El resultado sugiere que el patron predictivo esta dominado por dinamicas de commodities, memoria del oro y variables cambiarias/monetarias.
- La caja gris debe considerarse un benchmark interpretable, no el modelo final, hasta probar mas especificaciones.

## Interpretacion

El Random Forest se interpreta como modelo predictivo no causal. Sus importancias ayudan a identificar variables que mejoran pronostico fuera de muestra, pero no prueban causalidad.

El modelo de caja gris separa:

- componente macro lineal: relacion economica interpretable;
- componente Random Forest residual: no linealidades, umbrales, interacciones por regimen y ruido estructurado.

Si el modelo de caja gris supera al Random Forest puro, hay evidencia practica de que la estructura economica aporta. Si no lo supera, el proyecto puede reportar que el patron predictivo esta dominado por no linealidades o por persistencia.

## Kaggle

Los notebooks en `notebooks/` estan listos para correr en Kaggle si se sube el repositorio como dataset o si se clona el repo dentro del notebook.

- `notebooks/01_umap_local_biplot_kaggle.ipynb`: genera el embedding UMAP, clusters y local biplot.
- `notebooks/02_random_forest_grey_box_kaggle.ipynb`: entrena Decision Tree, Random Forest y caja gris.
