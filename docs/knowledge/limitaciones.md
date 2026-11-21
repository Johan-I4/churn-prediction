# Limitaciones al interpretar una predicción
Fuente: sección «Limitaciones» del README del proyecto.

## Asociación, no causalidad
El modelo estima riesgo estadístico (asociación), no causalidad. Que un factor aumente el riesgo no garantiza que cambiarlo (por ejemplo, el tipo de contrato) elimine el abandono por sí solo.

## Datos estáticos y deriva
El conjunto de datos es público y estático. En un entorno productivo haría falta monitorear la deriva de datos (data drift) y reentrenar periódicamente.

## Dependencia de los supuestos de costos
El umbral de decisión 0.34 depende directamente de los supuestos de costos unitarios y de la tasa de éxito comercial de la oferta de retención.

## Variables correlacionadas entre sí
Cargo mensual (MonthlyCharges), cargo total (TotalCharges), antigüedad (tenure), número de servicios y el cargo mensual relativo a la antigüedad están relacionados entre sí, porque varias se calculan a partir de las otras. Por eso el signo de la contribución de una sola de ellas puede parecer contraintuitivo (por ejemplo, un cargo mensual alto que aparece reduciendo el riesgo). Conviene leer los factores en conjunto y no interpretar uno aislado.

## Alcance de la explicación
Las contribuciones por factor que muestra la API son exactas para la regresión logística: se calculan como coeficiente por valor de la variable. Son contribuciones al log-odds respecto al promedio del conjunto de entrenamiento y no son valores SHAP.
