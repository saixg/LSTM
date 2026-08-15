from pydantic import BaseModel

class GateStep(BaseModel):
    t: int
    token: str
    forget: float
    input: float
    output: float
    cell_state_norm: float
