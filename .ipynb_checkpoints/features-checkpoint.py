from datetime import timedelta
from feast import Entity, FeatureView, Field, ValueType, FeatureStore
from feast.types import Float32, String
from feast.infra.offline_stores.file_source import FileSource

entity = Entity(name="iris_id", value_type=ValueType.INT64, description="Iris flower ID")

iris_source = FileSource(
    path="data/iris_feast.parquet",  # or data/iris_feast.csv
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris_features = FeatureView(
    name="iris_features",
    entities=[entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    source=iris_source,
    online=True
)

