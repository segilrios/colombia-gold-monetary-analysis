# Estado del arte: Random Forest y modelos de caja gris

## Objetivo

Este documento resume literatura relevante para la siguiente rama del proyecto: usar Random Forest y modelos de caja gris para estudiar el precio del oro en Colombia, su relacion con variables macrofinancieras y su posible conexion con politica monetaria y periodos de gobierno.

## Dos significados de "caja gris"

En la literatura aparecen dos familias cercanas pero distintas:

- **Grey-box model**: modelo que combina estructura teorica o economica con componentes aprendidos desde datos. En nuestro caso, una ecuacion macro interpretable puede explicar una parte del retorno del oro y un Random Forest puede aprender no linealidades o residuos.
- **Grey system / grey forecasting models**: familia GM(1,1), Grey-Verhulst, Grey-Fourier y Markov-grey. Se usa mucho en series con informacion incompleta, muestras cortas o alta incertidumbre.

Para este proyecto conviene usar ambos enfoques de forma complementaria: caja gris como arquitectura economica + ML, y modelos grey como benchmarks de corto plazo o como modulo para ventanas ejecutivas con pocas observaciones.

## Hallazgos principales

### Random Forest para oro y metales preciosos

Breiman (2001) introduce Random Forest como ensamble de arboles con bootstrap y seleccion aleatoria de variables. Su atractivo para este proyecto es que captura interacciones no lineales sin imponer una forma funcional rigida.

Pierdzioch y Risse (2017) aplican Random Forest multivariado a retornos de oro, plata, platino y paladio. Su resultado central es que el enfoque multivariado supera a Random Forest univariados, lo que apoya incluir variables conectadas al oro y no solo rezagos del oro.

Cohen y Aiche (2023) usan indices bursatiles, VIX, commodities y rendimientos de bonos para predecir cambios del oro con arboles, Random Forest, GBRT y XGBoost. Su lectura para nuestro caso es directa: el oro debe modelarse con variables internacionales, incertidumbre, tasas, energia y commodities.

Sotelo Cenas, Gervacio Arteaga y Carranza Rios (2023) encuentran que Gradient Boosting y Random Forest estiman bien el precio del oro y permiten identificar variables incidentes. Aunque el contexto no es Colombia, refuerza el uso de modelos de arboles para ranking de variables.

El paper ya descargado en el repositorio, `gold-tree-based-models-ijisae-2023.pdf`, tambien justifica iniciar con Decision Tree, Random Forest, AdaBoost, Gradient Boosting y XGBoost como familia de modelos comparables.

### Random Forest en macroeconomia y politica monetaria

Goulet Coulombe (2024) propone Macroeconomic Random Forest, una adaptacion del Random Forest para modelos macro con parametros variables en el tiempo. Esta idea es muy compatible con el proyecto porque nuestras relaciones pueden cambiar por regimen: Uribe I/II, Santos I/II, Duque, Petro, choques de commodities o cambios de postura monetaria.

Forte (2024), para inflacion argentina, muestra que Random Forest puede ser comparable a expectativas de mercado y modelos econometricos tradicionales. Lo mas util para nosotros es la interpretacion no parametrica: la importancia de tasas, brechas cambiarias o inflacion puede depender del regimen.

Marin, Delgadillo y Von der Meden (2020/2023) combinan Random Forest Regression con funciones extraidas de un DSGE neokeynesiano para decision de banco central en Mexico. Esta es una referencia clave de caja gris aplicada a politica monetaria: no reemplaza la teoria, sino que la usa para estructurar el aprendizaje.

Bolhuis y Rayner (2020), en un working paper del FMI, proponen un marco de ML para nowcasting/forecasting macroeconomico con validacion fuera de muestra, ensambles e importancia de variables. Es una buena guia practica para no usar Random Forest como caja negra sin controles.

Buckmann, Joseph y Robertson (2021) muestran que la interpretabilidad en economia puede mejorar con permutation importance y Shapley values. Para nuestro proyecto, eso sugiere reportar importancias globales, SHAP por gobierno y cambios de importancia antes/despues de eventos monetarios.

### Modelos grey y caja gris para oro/metales

Gligoric et al. (2020) proponen un modelo hibrido estocastico-grey para precios de metales en mineria y revisan literatura de Grey Model GM(1,1), Fourier-GM y Markov-grey aplicados a oro y otros metales. La revision indica que los modelos grey son utiles cuando hay series cortas, incertidumbre o datos incompletos.

Manickam, Indrakala y Kumar (2023) proponen un Grey-Fourier-Markov para oro. La idea es corregir errores residuales de GM(1,1), GM(2,1) o Grey-Verhulst con Fourier y Markov. Esto es relevante para nuestro caso si queremos modelar cada gobierno como una ventana corta.

Pang et al. (2015) usan un grey-box model para prediccion financiera de cash-flow y muestran que un modelo no lineal de caja gris puede superar modelos lineales de panel. Aunque no es oro ni politica monetaria, es una referencia importante para justificar la arquitectura: una parte teorica estructurada y una parte flexible de datos.

## Implicaciones para nuestra base

La base contiene `Precio_Oro`, `TRM`, `TIPM`, `Precio_BRENT`, `DTF`, inflacion sin alimentos, energia, cafe y Bancolombia. La literatura sugiere no predecir solamente niveles del oro, sino construir tambien:

- retorno logaritmico del oro;
- diferencia o variacion porcentual de TRM, TIPM, DTF, Brent e inflacion;
- rezagos de 1, 3, 6 y 12 periodos;
- ventanas moviles de volatilidad;
- dummies de gobierno;
- postura monetaria: `tightening`, `easing`, `neutral`;
- eventos de politica monetaria fuertes;
- interacciones entre gobierno y variables macro.

## Diseno recomendado

### 1. Baselines obligatorios

Antes de Random Forest:

- naive/random walk;
- media movil;
- regresion lineal o ElasticNet;
- ARIMA/SARIMAX si la frecuencia final queda mensual.

Estos baselines evitan concluir que Random Forest mejora cuando solo esta copiando persistencia temporal.

### 2. Random Forest supervisado

Objetivos posibles:

- regresion: retorno del oro a 1, 3 o 6 meses;
- clasificacion: oro sube/baja;
- regimen: cluster UMAP como variable auxiliar o etiqueta de estado.

Validacion:

- particion temporal, nunca aleatoria;
- rolling window o expanding window;
- comparacion por gobierno;
- metricas: MAE, RMSE, directional accuracy y Diebold-Mariano contra baseline.

Interpretacion:

- permutation importance;
- SHAP global;
- SHAP por periodo presidencial;
- partial dependence o accumulated local effects para TRM, TIPM, DTF, Brent e inflacion.

### 3. Caja gris macro + Random Forest

Arquitectura sugerida:

```text
retorno_oro_t+h =
  componente_macro_lineal(TRM, TIPM, DTF, Brent, inflacion, energia)
  + RandomForest(residuos, rezagos, interacciones, regimenes)
```

Interpretacion:

- el bloque macro lineal conserva lectura economica;
- el Random Forest captura no linealidades, umbrales y cambios de regimen;
- las importancias del Random Forest se leen sobre el residuo, no sobre toda la relacion economica.

### 4. Macroeconomic Random Forest simplificado

Inspirado por Goulet Coulombe:

```text
retorno_oro = X_macro * beta_t + error
beta_t = funcion_de_estado(periodo, cluster_umap, postura_monetaria, volatilidad)
```

En una primera implementacion, se puede aproximar con:

- modelos separados por gobierno;
- Random Forest con variables de interaccion gobierno x macro;
- comparacion de importancias por cluster UMAP.

### 5. Grey models por ventanas cortas

Usar GM(1,1), Grey-Verhulst o Grey-Fourier como benchmarks dentro de cada gobierno o ventana de evento monetario. Esto puede ayudar cuando hay pocas observaciones por ventana y se quiere una prediccion parsimoniosa.

## Riesgos metodologicos

- Random Forest no extrapola bien tendencias fuera del rango observado; por eso conviene predecir retornos/diferencias, no niveles crudos.
- OOB error no reemplaza validacion temporal.
- La importancia de variables puede estar sesgada por correlacion entre predictores; usar permutation importance y SHAP.
- Una buena prediccion no implica causalidad. Para politica monetaria, DiD/event-study sigue siendo el modulo causal.
- Si se usan datos diarios con variables de politica que cambian mensualmente, hay que cuidar duplicacion de informacion y frecuencia mixta.
- No usar splits aleatorios porque filtran futuro hacia pasado.

## Plan de implementacion en esta rama

1. Crear modulo `src/gold_policy_analysis/supervised.py`.
2. Generar dataset supervisado con rezagos y retornos.
3. Entrenar baselines y Random Forest con validacion temporal.
4. Reportar metricas por horizonte y por gobierno.
5. Implementar caja gris: bloque lineal + Random Forest sobre residuos.
6. Generar interpretabilidad: permutation importance y SHAP si la dependencia queda disponible.
7. Conectar resultados con clusters UMAP ya generados.

## PDFs descargados

- `docs/literature/state_of_art/random-forests-breiman-2001.pdf`
- `docs/literature/state_of_art/macroeconomy-as-random-forest-2006.12724.pdf`
- `docs/literature/state_of_art/hybrid-stochastic-grey-metal-price-2020.pdf`
- `docs/literature/state_of_art/cash-flow-prediction-grey-box-2015.pdf`
- `docs/literature/state_of_art/gold-tree-based-models-ijisae-2023.pdf`

## Referencias clave

- Breiman (2001). Random Forests. https://doi.org/10.1023/A:1010933404324
- Pierdzioch & Risse (2017). Forecasting Precious Metal Returns with Multivariate Random Forests. https://ssrn.com/abstract=3160014
- Cohen & Aiche (2023). Forecasting gold price using machine learning methodologies. https://doi.org/10.1016/j.chaos.2023.114079
- Sotelo Cenas et al. (2023). Predictive Machine Learning models to estimate the price of gold. https://doi.org/10.32829/sej.v8i1.204
- Goulet Coulombe (2024). The macroeconomy as a random forest. https://doi.org/10.1002/jae.3030
- Forte (2024). Short-term inflation forecasting in Argentina with Random Forest models. https://ideas.repec.org/a/bcr/ensayo/v1y2024i84p141-159.html
- Marin et al. (2020/2023). Enhancing Central Bank Decision Making with Machine Learning. https://ssrn.com/abstract=4074127
- Bolhuis & Rayner (2020). Deus ex Machina? A Framework for Macro Forecasting with Machine Learning. https://www.imf.org/-/media/files/publications/wp/2020/english/wpiea2020045-print-pdf.pdf
- Buckmann et al. (2021). Opening the Black Box. https://doi.org/10.1007/978-3-030-66891-4_3
- Gligoric et al. (2020). Hybrid Stochastic-Grey Model to Forecast the Behavior of Metal Price in the Mining Industry. https://doi.org/10.3390/su12166533
- Manickam et al. (2023). A Novel Mathematical Study on the Predictions of Volatile Price of Gold Using Grey Models. https://ojs.wiserpub.com/index.php/CM/article/view/2389
- Pang et al. (2015). Cash Flow Prediction Using a Grey-Box Model. https://doi.org/10.1109/IConAC.2015.7313951
