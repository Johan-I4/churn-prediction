-- Consulta 10: Curva de bajas acumuladas por antigüedad
-- Pregunta: ¿En qué mes de vida se concentra la mayor parte de las bajas?
-- Qué hace: cuenta las bajas de cada mes de antigüedad en una CTE, las acumula con SUM() OVER (ORDER BY tenure) y muestra solo meses clave (1, 3, 6, 12, 24, 36, 48, 60 y 72) como porcentaje del total de bajas.
-- Cómo leerla: si a los 12 meses ya se acumula una gran parte de las bajas, la retención debe enfocarse en el primer año.
-- Técnicas: dos CTE encadenadas, función de ventana acumulada, subconsulta escalar.
WITH bajas_por_mes AS (
    SELECT tenure, SUM(churn) AS bajas
    FROM clientes
    GROUP BY tenure
),
acumulado AS (
    SELECT
        tenure,
        bajas,
        SUM(bajas) OVER (ORDER BY tenure) AS bajas_acumuladas
    FROM bajas_por_mes
)
SELECT
    tenure                                                                    AS meses_de_antiguedad,
    bajas_acumuladas,
    ROUND(100.0 * bajas_acumuladas / (SELECT SUM(bajas) FROM bajas_por_mes), 1) AS pct_de_todas_las_bajas
FROM acumulado
WHERE tenure IN (1, 3, 6, 12, 24, 36, 48, 60, 72)
ORDER BY tenure;
