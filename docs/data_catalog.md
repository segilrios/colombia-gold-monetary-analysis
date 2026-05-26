# Catalogo minimo de datos

## Datos crudos locales

Archivo detectado en `data/raw/`:

- `BD_Energía_Colombia.xlsx`
- Tamano aproximado: 800,600 bytes
- Hoja: `Hoja1`

Columnas principales detectadas:

- `Fecha`
- `Demanda_energética`
- `TRM`
- `TIPM`
- `Precio_BRENT`
- `Precio_Oro`
- `Precio_Café_CentUSD`
- `DTF`
- `Bancolombia_Price_USD`
- `Inflación_sin_alimentos`
- `Precio_bolsa_nacional_Energética`

## Nota de versionamiento

Los archivos en `data/raw/` no se suben a Git por defecto. El repositorio conserva solo el catalogo y la estructura para evitar versionar datos crudos pesados o sensibles.

## Uso esperado

La base permite construir un panel temporal con precio del oro, variables monetarias, tasa de cambio, energia, petroleo, cafe e inflacion. Para el proyecto, la variable objetivo inicial sera `Precio_Oro`, y las variables de politica/entorno macro se usaran como predictores, tratamientos o controles segun el modulo.
