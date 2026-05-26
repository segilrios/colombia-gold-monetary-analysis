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

## Estado

Estructura inicial del proyecto.
