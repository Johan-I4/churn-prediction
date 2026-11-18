## Resultados

Datos: 7043 clientes, churn global 26.5%. Split estratificado 80/20 (semilla 42).

### Validación cruzada (5 folds, solo train; media ± desviación)

| Modelo | AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Regresión logística | 0.8488 ± 0.0113 | 0.6755 ± 0.0295 | 0.5304 ± 0.0394 | 0.5935 ± 0.0295 | 0.8076 ± 0.0116 |
| Random Forest | 0.8440 ± 0.0099 | 0.6727 ± 0.0332 | 0.4890 ± 0.0223 | 0.5660 ± 0.0230 | 0.8010 ± 0.0112 |
| Gradient Boosting (HistGB) | 0.8366 ± 0.0056 | 0.6369 ± 0.0244 | 0.5084 ± 0.0212 | 0.5652 ± 0.0203 | 0.7925 ± 0.0098 |
| Baseline (clase mayoritaria) | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7346 ± 0.0001 |

Modelo elegido por AUC en validación cruzada: **Regresión logística**.

### Umbral de decisión

Supuestos: un cliente perdido vale 500, una oferta de retención cuesta 50 y retiene al cliente con probabilidad 0.3. El umbral que maximiza el beneficio neto (sobre predicciones out-of-fold) es **0.34**.

![Curva precision-recall](reports/precision_recall.png)

### Test (usado una sola vez)

| Métrica | Valor |
|---|---|
| Accuracy del baseline (clase mayoritaria) | 0.7346 |
| AUC | 0.8468 |
| Precision (umbral 0.34) | 0.5620 |
| Recall (umbral 0.34) | 0.7273 |
| F1 (umbral 0.34) | 0.6340 |
| Matriz de confusión (TN/FP/FN/TP) | 823 / 212 / 102 / 272 |
| Beneficio neto: modelo vs. contactar a todos | 16600 vs. -14350 |
