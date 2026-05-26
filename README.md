# gold-policy-colombia-biplot-umap

Repositorio de investigacion para analizar la relacion entre el precio del oro y las politicas monetarias de Colombia en diferentes ventanas ejecutivas.

## Nombres sugeridos para el repo

- `gold-policy-colombia-biplot-umap`
- `colombia-gold-monetary-windows`
- `gold-price-policy-colombia-ml`
- `oro-politica-monetaria-colombia`

Nombre recomendado: `gold-policy-colombia-biplot-umap`, porque comunica el activo, el enfoque institucional, el pais y dos tecnicas centrales del proyecto.

## Idea central

El proyecto estudia si los cambios en el precio del oro tienen relacion con decisiones, ciclos o ventanas de politica monetaria en Colombia. La investigacion combina exploracion multivariada, reduccion de dimensionalidad, modelos predictivos y enfoques causales para comparar periodos ejecutivos.

## Metodologia propuesta

- Local biplot para explorar asociaciones entre variables macroeconomicas, precio del oro y periodos de gobierno.
- UMAP para detectar estructuras no lineales y agrupamientos temporales.
- Difference-in-differences para evaluar cambios alrededor de eventos o ventanas de politica monetaria.
- Arboles de decision para identificar reglas interpretables asociadas a movimientos del oro.
- Modelos de caja gris para combinar estructura economica con flexibilidad predictiva.

## Estructura

```text
data/
  raw/          Datos originales sin modificar.
  interim/      Datos en transformacion.
  processed/    Datos listos para modelado.
  external/     Fuentes externas y series auxiliares.
notebooks/      Exploracion, prototipos y analisis reproducibles.
src/
  data/         Carga, limpieza y validacion de datos.
  features/     Construccion de variables y ventanas ejecutivas.
  models/       Arboles, caja gris y modelos comparativos.
  causal/       Difference-in-differences y disenos cuasi-experimentales.
  visualization/ Graficas, biplots, UMAP y reportes visuales.
reports/
  figures/      Figuras finales.
  tables/       Tablas finales.
docs/           Diseno metodologico y notas de investigacion.
configs/        Parametros de ejecucion.
scripts/        Comandos reproducibles del proyecto.
tests/          Pruebas de funciones criticas.
```

## Pregunta de investigacion

Existe una relacion estadistica, predictiva o causal entre el precio del oro y las politicas monetarias de Colombia cuando se observan diferentes ventanas ejecutivas?

## Documentacion inicial

- `docs/state_of_art.md`: estado del arte minimo.
- `docs/state_of_art_random_forest_grey_box.md`: estado del arte para Random Forest y caja gris.
- `docs/data_catalog.md`: inventario de la base cruda local.
- `docs/methodology_roadmap.md`: ruta metodologica inicial.
- `docs/implementation_feature_sergio.md`: primera implementacion UMAP/local biplot/clustering/DiD.
- `docs/literature/`: PDFs primarios y referencias abiertas descargadas.

## Primer flujo analitico

```bash
pip install -r requirements.txt
python scripts/run_policy_umap_biplot.py --config configs/project.yaml
python scripts/run_random_forest_grey_box.py --config configs/project.yaml
python scripts/run_exhaustive_random_forest_grey_box.py --config configs/project.yaml --preset quick
```

Este flujo carga la base cruda local, genera embedding UMAP, clusters, local biplot por cluster y una tabla inicial para contrastes tipo DiD/event-study alrededor de cambios fuertes en la tasa de politica monetaria.
El segundo comando entrena Decision Tree, Random Forest y un modelo de caja gris con bloque macro lineal mas Random Forest sobre residuos.

Figuras principales:

- `reports/figures/policy_umap_local_biplot_clusters.png`
- `reports/figures/policy_umap_by_executive_period.png`
- `reports/figures/random_forest_grey_box_predictions_h1.png`

Notebooks Kaggle:

- `notebooks/01_umap_local_biplot_kaggle.ipynb`
- `notebooks/02_random_forest_grey_box_kaggle.ipynb`

## Estado

Estructura inicial del proyecto, documentacion minima y literatura base.
