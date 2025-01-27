from zenml import Model, pipeline

@pipeline(
    model=Model(
        name="bitcoin_price_predictor"
    )
)
def ml_pipeline():