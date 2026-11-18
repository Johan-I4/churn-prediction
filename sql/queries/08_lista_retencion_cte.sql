-- Consulta 08: Lista de clientes activos a contactar para retención
-- Pregunta: ¿A qué clientes que siguen activos conviene ofrecerles una oferta primero?
-- Qué hace: con una CTE toma solo a los activos (churn = 0) y con otra les suma un punto por cada factor de riesgo (mes a mes, 12 meses o menos, fibra, cheque electrónico, sin soporte técnico); lista a los de 4 o 5 puntos.
-- Cómo leerla: es una regla simple hecha a mano; el modelo de machine learning del proyecto hace lo mismo con probabilidades y es más preciso.
-- Técnicas: CTE (WITH) encadenadas, JOIN de 3 tablas, WHERE, suma de condiciones, ORDER BY y LIMIT.
WITH activos AS (
    SELECT
        c.customer_id, c.tenure,
        f.contract, f.payment_method, f.monthly_charges,
        s.internet_service, s.tech_support
    FROM clientes AS c
    INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
    INNER JOIN servicios   AS s ON s.customer_id = c.customer_id
    WHERE c.churn = 0
),
puntuados AS (
    SELECT
        *,
        (contract = 'Month-to-month')
        + (tenure <= 12)
        + (internet_service = 'Fiber optic')
        + (payment_method = 'Electronic check')
        + (tech_support = 'No')          AS puntos_riesgo
    FROM activos
)
SELECT
    customer_id, puntos_riesgo, tenure, contract,
    internet_service, payment_method, monthly_charges
FROM puntuados
WHERE puntos_riesgo >= 4
ORDER BY puntos_riesgo DESC, monthly_charges DESC
LIMIT 15;
