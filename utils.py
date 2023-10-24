from pydantic import BaseModel, Field

class PCAParameters(BaseModel):
    n_components: int = Field(description='number of components to keep')
