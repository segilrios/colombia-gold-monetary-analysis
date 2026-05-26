# Ruta metodologica minima

## 1. Preparacion de datos

- Validar frecuencia temporal de la base.
- Estandarizar nombres de columnas.
- Crear transformaciones: retornos del oro, variaciones porcentuales, rezagos, volatilidad movil y diferencias.
- Definir ventanas ejecutivas y eventos de politica monetaria.

## 2. Analisis exploratorio

- Series temporales de `Precio_Oro`, `TRM`, `TIPM`, `DTF`, inflacion y Brent.
- Correlaciones moviles por ventana ejecutiva.
- Deteccion de rupturas o cambios de regimen.

## 3. UMAP y local biplot

- Construir matriz de variables macro normalizadas.
- Proyectar observaciones temporales con UMAP.
- Interpretar regiones locales con local biplot.
- Colorear el embedding por ventana ejecutiva, postura monetaria y tendencia del oro.

## 4. Modelos predictivos

- Linea base: regresion y arbol de decision.
- Modelos comparativos: Random Forest, Gradient Boosting o XGBoost.
- Validacion temporal, no aleatoria.
- Metricas: MAE, RMSE, directional accuracy y estabilidad por ventana.

## 5. Difference-in-differences

- Definir tratamiento: cambio de postura monetaria, evento de tasa, ventana presidencial o shock institucional.
- Definir comparacion temporal y controles.
- Probar supuestos de tendencias paralelas cuando aplique.
- Estimar efectos heterogeneos por ventana.

## 6. Caja gris

- Separar componente estructural macroeconomico y componente flexible.
- Usar variables teoricas como restricciones o bloque explicativo.
- Comparar interpretabilidad y desempeno frente a modelos puramente de machine learning.

## 7. Reporte

- Tablas de datos y variables.
- Figuras de UMAP/local biplot.
- Resultados predictivos.
- Resultados DiD.
- Discusion sobre relacion con politica monetaria colombiana.
