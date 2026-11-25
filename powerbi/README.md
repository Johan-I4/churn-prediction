# Mini dashboard de churn en Power BI Desktop

Una página sobre el proyecto de churn: cuánto se pierde, dónde se concentra y a quién conviene contactar primero. Los datos salen de un script del repo (`src/export_dashboard_data.py`), probado con `pytest`; el dashboard solo los muestra.

## Estado de validación (2026-11-24)

| Parte | Estado |
|---|---|
| Script de exportación y sus 23 pruebas (`tests/test_export_dashboard_data.py`) | Comprobado (2026-11-24): 23/23 pasan |
| Cifras de los CSV contra el README (7,043 clientes, 26.54 %, segmentos) | Comprobado (2026-11-24) con pandas, pytest y Power BI |
| Medidas DAX (`medidas.dax`) y Power Query (`consultas.m`) | Comprobado (2026-11-24) en Power BI Desktop (`churn_dashboard.pbix`) |
| Cómo se ve el dashboard, colores, legibilidad y archivo `.pbix` (451 KB) | Comprobado (2026-11-24): publicado en `captura_dashboard.png` y `churn_dashboard.pbix` |

## Archivos

```text
powerbi/
├── README.md                 # esta guía
├── medidas.dax               # medidas DAX completas
├── consultas.m               # Power Query (M) de las 2 tablas
├── captura_dashboard.png     # captura validada del dashboard (1280 × 720)
├── churn_dashboard.pbix      # archivo de Power BI Desktop (451 KB)
└── data/                     # CSV generados (no se suben a GitHub)
    ├── clientes.csv
    └── resumen_modelo.csv
```

## Modelo de datos

Dos tablas, **sin relaciones** (a propósito: `resumen_modelo` es una tabla de parámetros de una fila y no debe filtrarse con los segmentadores).

**`clientes`** (7,043 filas, una por cliente)

| Columna | Qué es |
|---|---|
| `customer_id` | ID original del CSV |
| `Contract`, `InternetService`, `PaymentMethod` | Valores originales del dataset (en inglés, como en el README y el SQL) |
| `tenure`, `MonthlyCharges`, `TotalCharges` | Antigüedad en meses y cargos (`TotalCharges` ya limpio: 11 vacíos → 0) |
| `tenure_range` / `tenure_range_order` | `0-6 meses`, `7-12 meses`, `13-24 meses`, `25-48 meses`, `49-72 meses` (mismos cortes que la consulta SQL 03) y su orden 1–5 |
| `churn` | 1 = se dio de baja (dato real), 0 = no |
| `split` | `train` (5,634) o `test` (1,409): la misma partición de `src/train.py` |
| `churn_probability` | Probabilidad del modelo (sin redondear) |
| `predicted_churn` | 1 si la probabilidad ≥ umbral guardado en el modelo (0.34) |
| `risk_level` / `risk_order` | Bajo (p < 0.17), Medio (0.17 ≤ p < 0.34), Alto (0.34 ≤ p < 0.68), Muy alto (p ≥ 0.68) y su orden 1–4 |
| `top_factor_*` | Factor que más sube el riesgo del cliente (`src/explain.py`, contribución exacta de la regresión logística) |

**`resumen_modelo`** (1 fila): `threshold`, `n_train`, `n_test` y las métricas del test (`test_roc_auc`, `test_precision`, `test_recall`, `test_f1`, `test_accuracy`, matriz de confusión), tomadas de `reports/metrics.json`.

### Decisiones y límites

- **Leakage.** El modelo vio el 80 % de los clientes. Los 7,043 están puntuados para poder priorizar, pero **AUC, precision y recall solo salen del test**. Antes de exportar, el script recalcula esas tres métricas con las filas `split = test` y se detiene si no coinciden con `metrics.json` (así detecta una semilla o un `--test-size` distinto).
- **Churn %, clientes e ingreso en bajas** son datos reales de los 7,043 clientes; no dependen del modelo.
- **Niveles de riesgo.** Los cortes (0.5× y 2× el umbral) son una decisión de diseño anclada al umbral, no un resultado estadístico. Para las filas `train` las probabilidades son *in-sample*: sirven para priorizar, no para medir desempeño.
- **Factor principal.** Las contribuciones numéricas se miden contra el promedio del train y variables correlacionadas (cargo mensual, cargo total, antigüedad) pueden dar signos contraintuitivos. Son asociaciones, no causas. Si algún día el modelo deja de ser regresión logística, esas columnas salen vacías.
- **Moneda.** El dataset no declara moneda; el dashboard muestra el ingreso sin símbolo.

## Paso a paso (PowerShell y Power BI Desktop)

Supuesto: Power BI Desktop en español, versión reciente de 2026. Entre paréntesis va el nombre en inglés por si tu interfaz está en ese idioma.

### 1. Generar los CSV

```powershell
cd C:\ruta\a\churn-prediction
.\.venv\Scripts\Activate.ps1
python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv   # solo si no existe models\churn_pipeline.joblib
python -m src.export_dashboard_data
```

Tarda alrededor de un minuto. Debe terminar con `Clientes: 7,043  |  churn global: 26.54 %` y `Comprobación OK`.
Verifica la codificación (en Windows PowerShell 5.1 hace falta `-Encoding UTF8` o verás `AntigÃ¼edad`):

```powershell
Get-Content powerbi\data\clientes.csv -Encoding UTF8 -TotalCount 2
```

### 2. Parámetro con la ruta

Inicio > Transformar datos (*Transform data*) > Administrar parámetros (*Manage parameters*) > Nuevo parámetro:
nombre `RutaDatos`, tipo Texto, valor actual `C:\ruta\a\churn-prediction\powerbi\data\` (con `\` al final).

### 3. Importar las dos tablas

En el Editor de Power Query: Nueva consulta > Consulta en blanco (*Blank query*) > Editor avanzado (*Advanced editor*). Pega la consulta `clientes` de [`consultas.m`](consultas.m), pulsa Listo y renómbrala `clientes`. Repite con `resumen_modelo`. Cierra y aplica (*Close & Apply*).

### 4. Ordenar columnas

Selecciona `clientes[tenure_range]` > Herramientas de columna > Ordenar por columna > `tenure_range_order`. Igual con `risk_level` > `risk_order`. Oculta `tenure_range_order` y `risk_order`.

### 5. Crear las medidas

Selecciona la tabla `clientes` > Inicio > Nueva medida (*New measure*) y pega cada bloque de [`medidas.dax`](medidas.dax) por separado. Aplica el formato indicado en el comentario `Formato:` de cada una (Herramientas de medida).

### 6. Armar la página (1280 × 720, 16:9)

Archivo > Opciones > Carga de datos: desactiva "Fecha y hora automáticas" (no hay fechas).

| Zona (x, y, ancho × alto) | Visual | Campos |
|---|---|---|
| (24, 16, 760 × 48) | Cuadro de texto: título «Churn de clientes Telco» y subtítulo «7,043 clientes · modelo: regresión logística, umbral 0.34» | |
| (800, 16, 220 × 56) | Segmentador (*Slicer*), lista desplegable | `clientes[Contract]` |
| (1036, 16, 220 × 56) | Segmentador, lista desplegable | `clientes[InternetService]` |
| (24, 84, 400 × 84) | Tarjeta | `[Clientes]` |
| (440, 84, 400 × 84) | Tarjeta | `[Churn %]` |
| (856, 84, 400 × 84) | Tarjeta | `[Ingreso mensual en bajas]` (título: «Ingreso mensual en cuentas dadas de baja») |
| (24, 184, 400 × 260) | Columnas agrupadas | Eje `Contract`; Valores `[Churn %]` |
| (440, 184, 400 × 260) | Columnas agrupadas | Eje `tenure_range`; Valores `[Churn %]` |
| (856, 184, 400 × 260) | Barras agrupadas (horizontal), orden descendente por `[Churn %]` | Eje `PaymentMethod`; Valores `[Churn %]` |
| (24, 460, 560 × 244) | Columnas agrupadas | Eje `risk_level`; Valores `[Clientes]`; en Información sobre herramientas agrega `[% de clientes por nivel]` |
| (600, 460, 656 × 150) | 3 tarjetas en fila | `[AUC (test)]`, `[Precision (test)]`, `[Recall (test)]` |
| (600, 616, 656 × 88) | Cuadro de texto (nota) | «Desempeño: solo el 20 % de clientes que el modelo no vio (n = 1,409). Los niveles de riesgo puntúan a los 7,043 clientes para priorizar, no para medir desempeño.» |

**Colores** (todos con contraste suficiente sobre fondo claro; comprueba el resultado en tu captura):

| Uso | Color |
|---|---|
| Fondo de página | `#F7F8FA` |
| Texto | `#1F2937` |
| Barras de churn | `#C8553D` (terracota) |
| Tarjetas y bordes | blanco `#FFFFFF`, borde `#D9DEE5` |
| Riesgo Bajo / Medio / Alto / Muy alto | `#8FB996` / `#F2C14E` / `#E8833A` / `#B23A3A` |

Detalles: etiquetas de datos activadas, formato del eje en `0%`, sin cuadrículas pesadas. En las 3 gráficas de churn, Panel de análisis > Línea de valor constante = `0.2654` (churn global; línea gris discontinua con etiqueta «Global»). Para que los segmentadores no afecten las tarjetas de desempeño no hace falta configurar nada: `resumen_modelo` no tiene relaciones.

### 7. Guardar y capturar

Guarda como `powerbi\churn_dashboard.pbix`. Con la página abierta: Archivo > Exportar > Exportar a PDF, o la herramienta Recortes de Windows (`Win + Shift + S`), y guarda la imagen como `powerbi\captura_dashboard.png`.

## Lista de verificación contra el README

Comprobado en Power BI Desktop el 2026-11-24 (filas 1–6, 10 y 12 visibles en `captura_dashboard.png`; filas 7–9, 11 y 13 validadas con segmentadores y tabla auxiliar de verificación).

| # | Dónde mirar | Debe salir | Origen de la cifra | Estado |
|---|---|---|---|---|
| 1 | Tarjeta `[Clientes]` | 7,043 | README | [x] Comprobado |
| 2 | Tarjeta `[Churn %]` | 26.54 % | README | [x] Comprobado |
| 3 | Tarjeta `[Ingreso mensual en bajas]` | 139,130.85 | CSV (suma de `MonthlyCharges` con churn = 1) | [x] Comprobado |
| 4 | Churn % por `Contract` | Month-to-month 42.71 %, One year 11.27 %, Two year 2.83 % | SQL 02 | [x] Comprobado |
| 5 | Churn % por `tenure_range` | 52.94 %, 35.89 %, 28.71 %, 20.39 %, 9.51 % (en ese orden) | SQL 03 | [x] Comprobado |
| 6 | Churn % por `PaymentMethod` | Electronic check 45.29 % (el más alto), Mailed check 19.11 %, Bank transfer 16.71 %, Credit card 15.24 % | CSV (el README solo dice que el cheque electrónico es el más alto) | [x] Comprobado |
| 7 | Segmentador `Contract` = Month-to-month, medida `[% ingreso mensual en bajas]` | 46.97 % | README | [x] Comprobado |
| 8 | Contract = Month-to-month + InternetService = Fiber optic, `[Churn %]` | 54.61 % | README | [x] Comprobado |
| 9 | Contract = Month-to-month, gráfica por `PaymentMethod`: Electronic check | 53.73 % | README | [x] Comprobado |
| 10 | Tarjetas de test | AUC 0.8468, Precision 56.20 %, Recall 72.73 %, `[Clientes en test]` 1,409 | README y `metrics.json` | [x] Comprobado |
| 11 | Tabla con `[Precision (test, recalculada)]`, `[Recall (test, recalculada)]`, `[Clientes test (recalculado)]` | 56.20 %, 72.73 %, 1,409 (iguales a la fila 10) | Recalculo con `split = test` | [x] Comprobado |
| 12 | Gráfica de riesgo | Bajo 3,351 · Medio 1,324 · Alto 1,744 · Muy alto 624 (suman 7,043) | Modelo entrenado (`random_state=42`) | [x] Comprobado |
| 13 | `[Clientes marcados por el modelo]` | 2,368 (= Alto + Muy alto) | Idem fila 12 | [x] Comprobado |

Si la fila 10 y la 11 no coinciden, el dashboard está mezclando filas de train: revisa que las medidas usen `split = "test"`.

## ¿PBIX o PBIP?

- **`.pbix`**: un solo archivo binario que incluye los datos. Con 7,043 filas pesa **451 KB** (ya medido y versionado en este repositorio). Súbelo solo si queda en pocos MB y acepta que cada guardado es un binario nuevo en el historial de Git.
- **PBIP (proyecto de Power BI)**: carpeta de texto (modelo semántico y reporte como archivos), pensada para Git: los cambios se ven en un diff y las revisiones son legibles. Archivo > Guardar como > Proyecto de Power BI (*Power BI Project*); puede requerir activar su función en Opciones > Características en vista previa (*Preview features*) según tu versión. Conviene si vas a seguir editando el dashboard.
- **Recomendación**: para un portafolio, el `.pbix` pequeño más la captura PNG es lo más simple. Si prefieres revisar cambios con Git, usa PBIP.
- **Lo que no pude verificar**: si PBIP es hoy estable o sigue en vista previa (la documentación de Microsoft que consulté no lo indica), qué versión exacta de Power BI Desktop lo requiere, y cómo se comporta con tu ruta `RutaDatos`. Lo que sí dice esa documentación: Power BI Desktop crea un `.gitignore` con `**/.pbi/localSettings.json` y `**/.pbi/cache.abf` (ya están en el `.gitignore` del repo), usa CRLF (configura `git config core.autocrlf true`), no se puede guardar en OneDrive/SharePoint y en Windows conviene una ruta corta (límite de 260 caracteres).
- Fuente: [Power BI Desktop projects (Microsoft Learn)](https://learn.microsoft.com/power-bi/developer/projects/projects-overview).

## Si algo falla

| Síntoma | Causa probable |
|---|---|
| `No encontré el modelo` | Falta entrenar: `python -m src.train` |
| `ExportConsistencyError ... roc_auc` | `--seed` o `--test-size` distintos a los del entrenamiento, o `metrics.json` no es del modelo actual: reentrena y vuelve a exportar |
| Letras raras (`AntigÃ¼edad`) | El archivo se abrió como ANSI: en Power Query verifica `Encoding = 65001` |
| `Expression.Error: ... RutaDatos` | El parámetro no existe o no termina en `\` |
| Números mal (`0,34` o 34 en vez de 0.34) | La consulta perdió la cultura `"en-US"` en `Table.TransformColumnTypes` |
