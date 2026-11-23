// =====================================================================================
// Power Query (M) del dashboard de churn.
// Cómo usarlo (detalle en powerbi/README.md):
//   1. Inicio > Transformar datos > Administrar parámetros > Nuevo parámetro:
//        Nombre: RutaDatos   Tipo: Texto   Valor actual: C:\ruta\a\churn-prediction\powerbi\data\
//      (debe terminar en \ ; es la ÚNICA cosa que cambia de una computadora a otra)
//   2. Nueva consulta > Consulta en blanco > Editor avanzado: pega la consulta `clientes`
//      y renómbrala `clientes`. Repite con `resumen_modelo`.
// Los CSV están en UTF-8 sin BOM, separados por comas y con punto decimal. Por eso se fija
// Encoding=65001 y la cultura "en-US" al tipar: así no dependen del idioma de Windows.
// Estas consultas no se ejecutaron en Power BI al escribirse.
// =====================================================================================


// ---------------------------------------------------------------- Consulta: clientes
let
    Origen = Csv.Document(
        File.Contents(RutaDatos & "clientes.csv"),
        [Delimiter = ",", Columns = 20, Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    Encabezados = Table.PromoteHeaders(Origen, [PromoteAllScalars = true]),
    Tipos = Table.TransformColumnTypes(
        Encabezados,
        {
            {"customer_id", type text},
            {"Contract", type text},
            {"InternetService", type text},
            {"PaymentMethod", type text},
            {"tenure", Int64.Type},
            {"tenure_range", type text},
            {"tenure_range_order", Int64.Type},
            {"MonthlyCharges", type number},
            {"TotalCharges", type number},
            {"churn", Int64.Type},
            {"split", type text},
            {"churn_probability", type number},
            {"predicted_churn", Int64.Type},
            {"risk_level", type text},
            {"risk_order", Int64.Type},
            {"top_factor_feature", type text},
            {"top_factor_label", type text},
            {"top_factor_value", type text},
            {"top_factor_text", type text},
            {"top_factor_contribution", type number}
        },
        "en-US"
    )
in
    Tipos


// ---------------------------------------------------------------- Consulta: resumen_modelo
let
    Origen = Csv.Document(
        File.Contents(RutaDatos & "resumen_modelo.csv"),
        [Delimiter = ",", Columns = 14, Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    Encabezados = Table.PromoteHeaders(Origen, [PromoteAllScalars = true]),
    Tipos = Table.TransformColumnTypes(
        Encabezados,
        {
            {"model_name", type text},
            {"threshold", type number},
            {"n_train", Int64.Type},
            {"n_test", Int64.Type},
            {"test_roc_auc", type number},
            {"test_precision", type number},
            {"test_recall", type number},
            {"test_f1", type number},
            {"test_accuracy", type number},
            {"test_tn", Int64.Type},
            {"test_fp", Int64.Type},
            {"test_fn", Int64.Type},
            {"test_tp", Int64.Type},
            {"sklearn_version", type text}
        },
        "en-US"
    )
in
    Tipos
