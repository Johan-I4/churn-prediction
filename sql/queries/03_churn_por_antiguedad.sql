-- Consulta 03: Churn por rango de antigüedad
-- Pregunta: ¿Los clientes nuevos abandonan más que los antiguos?
-- Qué hace: convierte tenure (meses) en 5 rangos con CASE, agrupa por rango y calcula la tasa de baja de cada uno. Solo usa la tabla clientes.
-- Cómo leerla: con el CSV real debe dar 52.94 % (0-6), 35.89 % (7-12), 28.71 % (13-24), 20.39 % (25-48) y 9.51 % (49-72); la tasa baja al aumentar la antigüedad.
-- Técnicas: CASE, GROUP BY sobre una expresión, ORDER BY MIN para ordenar los rangos.
SELECT
    CASE
        WHEN tenure <= 6  THEN '0-6'
        WHEN tenure <= 12 THEN '7-12'
        WHEN tenure <= 24 THEN '13-24'
        WHEN tenure <= 48 THEN '25-48'
        ELSE '49-72'
    END                                 AS rango_meses,
    COUNT(*)                            AS clientes,
    ROUND(100.0 * AVG(churn), 2)        AS pct_churn
FROM clientes
GROUP BY rango_meses
ORDER BY MIN(tenure);
