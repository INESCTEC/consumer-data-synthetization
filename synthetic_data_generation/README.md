# Synthetic Data Generation Service

This service provides an API for generating synthetic data from locally trained
DoppelGANger models.

Model training is handled elsewhere: on demand from caller-supplied CSVs by the
separate `synthetic_model_training` service, or in batch from smart-meter
history by the standalone `services/synthetic_model_pretrainer` CLI. All of them
must point at the same `SYNTHETIC_MODEL_ROOT` folder or Docker volume, so a
model produced by any of them is immediately visible here without redeployment.

## Endpoints

### `GET /generate`

Generates `number_of_examples` synthetic sequences from a named model.

| parameter | meaning |
| --- | --- |
| `model_name` | the name `/models` published for the model (an alias for a pre-trained one - see [Pseudonymisation](#pseudonymisation)) |
| `user_id` | optional namespace (`models/<user_id>/<model_name>`) |
| `number_of_examples` | how many independent sequences to return |
| `hours` | length of each sequence; defaults to the length the model was trained on |
| `format` | `json` (default), `csv`, or `zip` of Parquet files |

Each example is an **independent draw**, not a segment of one longer series.
Timestamps are a regular grid at the model's own resolution: for a model
pre-trained on a real meter the grid starts at the moment of generation, and for
a model a user trained themselves it starts where their own training data does.

`hours` is validated against the horizons the model was certified for (see
`capabilities` in its metadata) and rejected otherwise. The generator is
recurrent and will roll out to any length, so without this check the API would
silently serve output whose quality was never verified.

Values are clipped to the physical bounds recorded at training time
(`value_floor`/`value_ceiling`), since the generator's output layer is unbounded
and would otherwise emit negative average power for a counter that cannot run
backwards.

### `GET /models`

Lists available models with a capability summary for each: device, parameter,
resolution, days of real data trained on, and verified generation horizons.
Enough to build a model picker without one `/model-info` call per model.

Models trained before the capability evaluation have no summary; treat a missing
`offered_hours` as "only the trained length is known good".

### `GET /model-info`

Training metadata for one model, including its capability evaluation, with the
provenance of a pre-trained model removed (see below).

### `GET /real-data`

The real series a model was trained on (`train_data.csv`), for comparing
generated output against it. Values are average power in W on the model's own
grid. `start` carries the same anonymised timestamp `/generate` uses, not the
date the series was recorded.

## Pseudonymisation

The pre-trained models ship with the platform and are trained on real smart
meters. Their folder name is the meter's own identifier and their metadata
records the source endpoint, the calendar span of the raw series and when
training and evaluation ran - enough to work back from generated data to the
household and period behind it, which is what synthetic data exists to avoid.

So for those models this service publishes nothing real (`anonymize.py`):

- the model is published as **`ResidentialConsumer_NN`**, and that alias is the
  only address it answers to. The real folder name is **not** accepted, so the
  mapping cannot be probed by guessing names;
- `device_id` is the same alias, and `source`, `raw_fetch`, `trained_at` and
  `capabilities.evaluated_at` are dropped;
- generated series are dated from the moment of generation.

Aggregate quality figures - days trained on, resolution, verified horizons and
the per-horizon metrics - are published unchanged: they say how far to trust the
output and identify nobody.

Models a user trained themselves are untouched: they carry no `device_id` or
`raw_fetch`, so they keep their name and their own timestamps.

### How the numbers are assigned

`NN` is a position in an ordering keyed by the deployment secret, so it says
nothing about the meter behind it - not even where it sits on disk. Models
offering at least one verified horizon are numbered first, so every model a
client can actually generate from gets a two-digit number: of the 146 models
currently shipped, the 28 usable ones are `ResidentialConsumer_00` through
`_27`, and the 118 that failed their quality evaluation - which a picker hides
anyway - run on to `_145`.

Because the numbers are positions, adding or removing models renumbers the
others. Pin `SYNTHETIC_ALIAS_SECRET` **and** the model set together if a name
has to survive a redeployment.

The secret lives in `$SYNTHETIC_MODEL_ROOT/.alias_secret`, created on first use
and shared by every replica on the volume. Set `SYNTHETIC_ALIAS_SECRET` to pin
the aliases across deployments that do not share it.

## Authentication

> [!IMPORTANT]
> Keycloak authentication is configured, but `auth.py` bypasses token
> verification for `GET` requests. Every endpoint in this service is a `GET`, so
> in practice **none of them require a token**.

## Features

-   **Synthetic Data Generation**: Generate realistic time-series data from pre-trained local models.
-   **Model Catalog**: List and inspect available local model folders.
-   **Configurable Requirements**: Switch between CPU and GPU dependencies via environment variables.

## Configuration

The service uses the following environment variable to determine which requirements to install and use:

- `SYNTHETIC_DATA_DEVICE`: Set to `CPU` (default) or `GPU`.

### Local Development

When running locally with `start_server.ps1`, the script will check this variable to install the appropriate `requirements.txt` or `requirementsGPU.txt`.

### Docker Deployment

The `docker-compose.yml` passes this variable as a build argument to the Dockerfile to ensure the correct dependencies are baked into the image.

## Running Locally

1. Uncomment the line in `start_server.ps1` that sets the device (optional):
   ```powershell
   $env:SYNTHETIC_DATA_DEVICE = "GPU"
   ```
2. Run the start script:
   ```powershell
   ./start_server.ps1
   ```

## Running with Docker (Standalone)

```powershell
./start_dockerized_server.ps1
```
