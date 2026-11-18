import pytest
from pydantic import ValidationError

from src.schemas import CustomerFeatures

VALID = CustomerFeatures.model_config["json_schema_extra"]["example"]


def test_valid_customer():
    assert CustomerFeatures(**VALID).tenure == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("Contract", "Three year"),
        ("tenure", -1),
        ("MonthlyCharges", -10.0),
        ("SeniorCitizen", 2),
        ("gender", "Otro"),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, field: value})


def test_missing_and_extra_fields_are_rejected():
    missing = {k: v for k, v in VALID.items() if k != "tenure"}
    with pytest.raises(ValidationError):
        CustomerFeatures(**missing)
    with pytest.raises(ValidationError):
        CustomerFeatures(**VALID, customerID="X")
