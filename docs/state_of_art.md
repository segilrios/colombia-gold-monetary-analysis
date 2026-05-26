# Estado del arte minimo

## Pregunta del proyecto

El proyecto busca estudiar si el precio del oro en Colombia se relaciona con la politica monetaria y con ventanas ejecutivas especificas. La estrategia combina exploracion interpretable, reduccion de dimensionalidad, modelos predictivos y disenos cuasi-experimentales.

## Paper principal

El paper ancla es:

Pérez-Rosero, D. A., Manrique-Cabezas, D. A., Triana-Martinez, J. C., Álvarez-Meza, A. M., & Castellanos-Dominguez, G. (2025). *An Explainable Framework Integrating Local Biplots and Gaussian Processes for Unemployment Rate Prediction in Colombia*. Computation, 13(5), 116. https://doi.org/10.3390/computation13050116

Este trabajo es central porque propone un marco explicable para Colombia que integra biplots locales y procesos gaussianos. La adaptacion natural del proyecto consiste en cambiar el dominio de desempleo hacia precio del oro, politica monetaria y ventanas ejecutivas.

PDF local:

- `docs/literature/primary/computation-13-00116-v2.pdf`

## Lineas de literatura relevantes

### 1. Local biplot y explicabilidad

Los local biplots extienden la interpretacion tipo biplot a mapas de baja dimension, permitiendo observar que variables explican regiones locales del embedding. Fukuyama (2021) los plantea como una forma de interpretar relaciones entre muestras y variables en representaciones de multidimensional scaling.

Uso en este proyecto:

- Interpretar regiones del mapa donde el precio del oro cambia de regimen.
- Identificar variables macro que dominan localmente: TRM, DTF, TIPM, inflacion, Brent o energia.
- Comparar ventanas ejecutivas mediante cargas locales.

### 2. UMAP para estructura no lineal

UMAP es un metodo de reduccion de dimensionalidad no lineal basado en geometria riemanniana y topologia algebraica. Es util para visualizar estructura local y global en datos de alta dimension.

Uso en este proyecto:

- Construir mapas de meses/trimestres segun similitud macrofinanciera.
- Detectar agrupamientos por periodos de politica monetaria.
- Evaluar si las ventanas ejecutivas se separan naturalmente en el espacio de variables.

### 3. Politica monetaria y commodities

El documento del Banco de la Republica sobre choques de politica monetaria y precios de commodities es clave para Colombia. El texto resalta que los precios de commodities afectan tasa de cambio, precios, ingreso nacional y balanza de pagos, y estudia efectos de choques monetarios sobre precios individuales como oro, carbon, niquel y petroleo.

Uso en este proyecto:

- Justificar que el oro no es solo una serie financiera, sino una variable conectada con transmision monetaria y sector externo.
- Usar SVAR o ventanas de eventos como contraste econometrico frente a modelos de machine learning.
- Definir controles externos: dolar, tasas, inflacion, Brent y ciclos globales.

### 4. Difference-in-differences

Callaway y Sant'Anna (2021) formalizan DiD con multiples periodos y adopcion escalonada. Es relevante si las ventanas ejecutivas o eventos de politica monetaria pueden definirse como tratamientos en distintos momentos.

Uso en este proyecto:

- Comparar periodos antes/despues de cambios de regimen monetario.
- Modelar efectos heterogeneos por ventana ejecutiva.
- Evitar interpretaciones debiles de TWFE cuando hay efectos dinamicos o heterogeneos.

### 5. Arboles de decision y modelos basados en arboles

La literatura reciente de prediccion del oro usa arboles de decision, Random Forest, AdaBoost, Gradient Boosting y XGBoost con indicadores tecnicos y macrofinancieros. Estos modelos son utiles como linea base interpretable y como comparacion predictiva.

Uso en este proyecto:

- Estimar reglas simples para movimientos de `Precio_Oro`.
- Extraer importancia de variables.
- Comparar reglas por ventana ejecutiva.

### 6. Modelos de caja gris

Los modelos de caja gris combinan estructura teorica con aprendizaje estadistico. Para este proyecto, la idea es imponer relaciones macro razonables, por ejemplo tasas, inflacion, TRM y commodities, y dejar que modelos flexibles capturen no linealidades residuales.

Uso en este proyecto:

- Separar componente economico estructural y componente no lineal.
- Comparar modelos puramente predictivos contra modelos con restricciones o variables teoricamente seleccionadas.
- Mantener interpretabilidad frente a un problema de politica economica.

## Brecha del proyecto

La literatura cubre por separado prediccion del precio del oro, politica monetaria y commodities, UMAP/local biplot, y DiD. La contribucion propuesta es integrar esas piezas en un analisis para Colombia con ventanas ejecutivas, usando una base macro local y documentando tanto patrones predictivos como evidencia cuasi-experimental.

## Referencias iniciales

- Pérez-Rosero et al. (2025). Local biplots y procesos gaussianos para desempleo en Colombia. https://doi.org/10.3390/computation13050116
- McInnes, Healy & Melville (2020). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. https://arxiv.org/abs/1802.03426
- Fukuyama (2021). Local biplots for multidimensional scaling. https://arxiv.org/abs/2008.02662
- Callaway & Sant'Anna (2021). Difference-in-Differences with multiple time periods. https://doi.org/10.1016/j.jeconom.2020.12.001
- Banco de la Republica (2011). The effect of monetary policy on commodity prices. https://www.banrep.gov.co/sites/default/files/publicaciones/pdfs/borra685.pdf
- Baser, Saini & Baser (2023). Gold Commodity Price Prediction Using Tree-based Prediction Models. https://www.ijisae.org/index.php/IJISAE/article/view/2481
