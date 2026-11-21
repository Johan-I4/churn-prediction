# Cómo se decide y qué tan confiable es el modelo
Fuente: reports/results.md y reports/metrics.json del proyecto.

## Umbral de decisión por costos
El umbral de decisión es 0.34, no el corte estándar de 0.50. Se eligió maximizando el beneficio neto con tres supuestos: un cliente perdido vale 500, una oferta de retención cuesta 50 y retiene al cliente con probabilidad 0.30. El umbral depende de esos supuestos: si cambian, el umbral óptimo cambia.

## Modelo elegido y métricas
El modelo elegido por AUC en validación cruzada (5 folds) fue la regresión logística, con AUC de 0.8488 ± 0.0113 en validación cruzada. En el conjunto de prueba, usado una sola vez, obtuvo AUC de 0.8468, recall de 72.73 % y precision de 56.20 % con el umbral 0.34.

## Beneficio neto frente a contactar a todos
En el conjunto de prueba, con los supuestos de costos, contactar solo a los clientes que el modelo marca como riesgo da un beneficio neto de 16,600, frente a -14,350 si se contactara a todos los clientes.

## Qué significa precision y recall en este caso
Con recall de 72.73 % el modelo detecta unos 7 de cada 10 clientes que realmente abandonan. Con precision de 56.20 %, poco más de la mitad de los clientes marcados como riesgo abandona de verdad; el resto son falsas alarmas que cuestan una oferta de retención.
