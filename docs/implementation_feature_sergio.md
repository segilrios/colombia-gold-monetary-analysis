# Implementacion inicial en `feature/sergio`

## Referencia revisada

Repositorio externo:

- https://github.com/UN-GCPDS/Unemployment-Rate-Prediction/tree/main

La referencia del primer articulo organiza el trabajo en notebooks de enfoque supervisado y no supervisado. El notebook no supervisado integra UMAP, KMeans, PCA local y transformaciones afines para construir local biplots. En este repositorio se implementa una version propia y modular para el caso de precio del oro y politica monetaria en Colombia.

## Flujo implementado

Script principal:

```bash
python scripts/run_policy_umap_biplot.py --config configs/project.yaml
```

El flujo realiza:

- Carga de `data/raw/BD_Energía_Colombia.xlsx`.
- Normalizacion de nombres de columnas.
- Asignacion de periodos ejecutivos de Colombia.
- Clasificacion de postura monetaria por cambios en `TIPM`.
- Embedding UMAP de variables macrofinancieras.
- Clustering KMeans sobre el embedding.
- Local biplot por cluster mediante PCA local y ajuste afin hacia el espacio UMAP.
- Diseno inicial tipo event-study/DiD alrededor de cambios fuertes en `TIPM`.

## Salidas esperadas

- `data/processed/policy_umap_embedding.csv`
- `data/processed/policy_event_did_design.csv`
- `reports/tables/cluster_profiles.csv`
- `reports/tables/local_biplot_loadings.csv`
- `reports/tables/policy_event_contrasts.csv`
- `reports/figures/policy_umap_local_biplot_clusters.png`
- `reports/figures/policy_umap_by_executive_period.png`

## Primera corrida local

La primera corrida sobre `BD_Energía_Colombia.xlsx` produjo:

- Observaciones usadas: 9,283.
- Rango temporal: 2000-01-01 a 2025-05-31.
- Metodo de embedding: UMAP.
- Clusters KMeans: 4.
- Eventos monetarios detectados por cambios fuertes en `TIPM`: 54.
- Periodos cubiertos: Pastrana, Uribe I, Uribe II, Santos I, Santos II, Duque y Petro.

## Nota sobre DiD

La base actual parece ser una serie temporal nacional. Un DiD causal requiere una unidad tratada y una unidad de control, o un panel con grupos comparables. Por eso el codigo actual crea una tabla lista para evolucionar hacia DiD y calcula contrastes descriptivos pre/post por evento monetario, pero no afirma causalidad todavia.
