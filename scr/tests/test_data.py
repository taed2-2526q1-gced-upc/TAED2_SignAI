# Test the data with great expectations

import great_expectations as gx
from src.config import INTERIM_DATA_DIR, ROOT_DIR

context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

datasource = context.data_sources.add_pandas(name = "GTSRB_images")

data_asset = datasource.add_parquet_asset(name="GTSRB_images_parquet", path=INTERIM_DATA_DIR / "GTSRB_cleaned.parquet")
batch = data_asset.add_batch_definition(name="GTSRB_images_data")

expectation_suite = gx.ExpectationSuite("GSTRB_images_data_validation")
context.suites.add_or_update(expectation_suite)

## Triar les expectatives que vulguem

expectation_suite.save()

# Validar la suit per cada batch
validator_definition = gx.ValidatorDefinition(
    name = "GTSRB_data_validator",
    data = batch_definition,
    suite = expectation_suite,
)
context.validations_definitions.add_or_update(validator_definition)

# Crear un checkpoint per executar la validació
action_list = [
    gx.checkpoint.UpdateDataDocsAction(name="update_data_docs"),
]
validator_definition = [validator_definition]

checkpoint = gx.Checkpoint(
    name = "GTSRB_data_checkpoint",
    validator_definitions = validator_definition,
    actions = action_list,
    results_format = "SUMMARY",
)

context.checkpoints.add_or_update(checkpoint)

# Executar el checkpoint
checkpoint_result = checkpoint.run()
context.view_validation_result(checkpoint_result)