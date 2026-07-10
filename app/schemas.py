from pydantic import BaseModel, ConfigDict, Field


class ClientFeatures(BaseModel):
    """Raw input fields, matching the original UCI dataset schema.
    Feature engineering (utilization ratios, delay trends, etc.) is applied
    server-side so API consumers only ever need to send raw client data."""

    LIMIT_BAL: float = Field(..., description="Credit limit (NT dollars)")
    SEX: int = Field(..., description="1=male, 2=female")
    EDUCATION: int = Field(..., description="1=grad school,2=university,3=high school,4=other")
    MARRIAGE: int = Field(..., description="1=married, 2=single, 3=other")
    AGE: int
    PAY_0: int = Field(..., description="Repayment status, most recent month")
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 46000,
                "BILL_AMT4": 44000, "BILL_AMT5": 42000, "BILL_AMT6": 40000,
                "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
                "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
            }
        }
    )


class PredictionResponse(BaseModel):
    default_probability: float
    default_prediction: int
    risk_tier: str
    threshold_used: float
    model_version: str