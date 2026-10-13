import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Response,
    status,
)
import zipfile
import io

import anonymize
from dgan_wrapper import generate, load_train_result
from fastapi.responses import JSONResponse
from auth import KEYCLOAK_CLIENT_ID
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Synthetic Data API",
    description="API for generating synthetic time-series data from locally pre-trained models.",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    swagger_ui_init_oauth={
        "clientId": KEYCLOAK_CLIENT_ID,
        "appName": "Synthetic Data Generation",
    },
    # No app-level auth dependency: every route below is GET, and verify_token
    # already bypasses GET requests on its own (see auth.py). An app-level
    # dependency here would also apply to the /control router mounted below,
    # which the orchestrator worker calls without a Keycloak token.
)

# Add CORS middleware to allow Swagger UI to communicate with Keycloak
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI-Effect Control Interface ---
try:
    from common import create_control_router, synthetic_data_generation_handlers

    app.include_router(
        create_control_router(synthetic_data_generation_handlers),
        prefix="/control",
    )
except ImportError as e:
    logger.warning(f"AI-Effect control interface not available: {e}")

MODEL_ROOT = Path(os.getenv("SYNTHETIC_MODEL_ROOT", "models")).resolve()


def _safe_child_path(*parts: str) -> Path:
    path = MODEL_ROOT.joinpath(*parts).resolve()
    if MODEL_ROOT != path and MODEL_ROOT not in path.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model path",
        )
    return path


def _model_path(model_name: str, user_id: Optional[str] = None) -> Path:
    """The folder behind the name a client was given.

    Models trained on real meters are addressable only by the alias /models
    published for them: their folder name is the meter's own identifier, so
    accepting it here would let a client confirm which meters exist simply by
    guessing names (see anonymize).
    """
    aliased = anonymize.resolve_alias(MODEL_ROOT, model_name)
    if aliased is not None:
        if user_id and aliased.parent != _safe_child_path(user_id):
            return _safe_child_path(user_id, model_name)
        return aliased

    if user_id:
        return _reject_if_pseudonymous(_safe_child_path(user_id, model_name), model_name)

    direct_path = _safe_child_path(model_name)
    if (direct_path / "model.pt").exists():
        return _reject_if_pseudonymous(direct_path, model_name)

    matches = [
        path.parent
        for path in MODEL_ROOT.rglob("model.pt")
        if path.parent.name == model_name
    ]
    if len(matches) == 1:
        return _reject_if_pseudonymous(matches[0], model_name)
    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple pre-trained models named '{model_name}' exist. "
                "Provide user_id or use a unique model folder name."
            ),
        )
    return direct_path


def _reject_if_pseudonymous(model_dir: Path, requested_name: str) -> Path:
    """Refuse a model that resolved by its real name rather than by its alias."""
    if anonymize.is_pseudonymous(anonymize.load_metadata(model_dir)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pre-trained model '{requested_name}' not found",
        )
    return model_dir


def _value_bounds(metadata: dict) -> tuple[Optional[float], Optional[float]]:
    """Physical bounds recorded at training time, if any.

    The generator's output is unbounded, so without these it emits negative
    average power for an import register that cannot run backwards.
    """
    return metadata.get("value_floor"), metadata.get("value_ceiling")


def _resolve_horizon_steps(
    metadata: dict, model_name: str, train_result, hours: int
) -> int:
    """Timesteps for a requested horizon, refused unless the model was measured
    to support it.

    The generator will happily roll out to any length, so without this check the
    API would silently serve output whose quality was never verified.
    """
    caps = metadata.get("capabilities") or {}

    # resolution_minutes is a top-level training field; capabilities repeats it.
    resolution = metadata.get("resolution_minutes") or caps.get("resolution_minutes")
    if not resolution:
        trained_hours = caps.get("trained_sequence_hours")
        if trained_hours:
            resolution = trained_hours * 60 // int(train_result.model.max_sequence_len)
    if not resolution:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This model has no recorded resolution, so 'hours' cannot be "
            "translated into timesteps. Omit 'hours' to use the trained length.",
        )

    offered = caps.get("offered_hours")
    if not offered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This model has no evaluated generation horizons. Omit 'hours' "
            "to use the length it was trained on.",
        )
    if hours not in offered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' does not support a {hours}h horizon. "
            f"Supported: {offered}.",
        )
    return hours * 60 // resolution


@app.get("/generate", tags=["Synthetic Data Generation"])
def generate_synthetic_data(
    model_name: str = Query("test_model", description="The name of the model"),
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
    number_of_examples: int = Query(1, description="Number of examples to generate"),
    hours: Optional[int] = Query(
        None,
        description="Length of each generated example, in hours. Defaults to the "
        "length the model was trained on. Only values listed in the model's "
        "capabilities.offered_hours are accepted - longer rollouts are possible "
        "architecturally but were measured to degrade.",
    ),
    format: str = Query(
        "json",
        pattern="^(json|csv|zip)$",
        description="Result format: 'json' (default, list of records), "
        "'csv' (single CSV attachment), or 'zip' (ZIP of Parquet files).",
    ),
):
    """
    Endpoint to generate synthetic data using a locally pre-trained DGAN model.
    """
    folder_path = _model_path(model_name, user_id)
    try:
        train_result = load_train_result(str(folder_path))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pre-trained model '{model_name}' not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {e}",
        )

    metadata = anonymize.load_metadata(folder_path)

    horizon_steps = None
    if hours is not None:
        horizon_steps = _resolve_horizon_steps(
            metadata, model_name, train_result, hours
        )

    floor, ceiling = _value_bounds(metadata)
    synthetic_dfs = generate(
        train_result,
        num_examples=number_of_examples,
        horizon_steps=horizon_steps,
        clip_min=floor,
        clip_max=ceiling,
        # For a model trained on a real meter the output is dated from now: its
        # own timestamps would otherwise say when the real series was recorded.
        start=anonymize.generation_start(metadata),
    )

    if not synthetic_dfs:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Echo back the name the client used, which for a pre-trained model is its
    # alias - never the folder name it resolved to.
    public_name = anonymize.public_name(MODEL_ROOT, folder_path, metadata)

    for i, synthetic_df in enumerate(synthetic_dfs):
        synthetic_df["example_id"] = i + 1
    combined_df = pd.concat(synthetic_dfs, ignore_index=True)

    if format == "json":
        # combined_df holds numpy dtypes (int64/float64) that plain json.dumps
        # can't serialize; to_json() handles that conversion natively.
        records = json.loads(combined_df.to_json(orient="records", date_format="iso"))
        return JSONResponse(content={"model_name": public_name, "data": records})
    elif format == "csv":
        return Response(
            combined_df.to_csv(index=False),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=synthetic_data_{public_name}.csv"
            },
        )
    else:
        zip_io = io.BytesIO()
        with zipfile.ZipFile(
            zip_io, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as temp_zip:
            for i, synthetic_df in enumerate(synthetic_dfs):
                zip_path = f"synthetic_data_{i+1}.parquet"
                temp_zip.writestr(zip_path, synthetic_df.to_parquet(index=False))
        return Response(
            zip_io.getvalue(),
            media_type="application/x-zip-compressed",
            headers={
                "Content-Disposition": f"attachment; filename=synthetic_data_{public_name}.zip"
            },
        )


def _capability_summary(model_dir: Path) -> dict:
    """The few metadata fields a model picker needs, read from metadata.json.

    anonymize.public_summary decides what may leave the service: for a model
    trained on a real meter the name and device id are aliases and the training
    timestamps are withheld, while the quality figures - which identify nobody -
    are passed through unchanged.
    """
    metadata = anonymize.load_metadata(model_dir)
    if not metadata:
        return {}
    return anonymize.public_summary(MODEL_ROOT, model_dir, metadata)


@app.get("/models", tags=["Synthetic Data Generation"])
async def list_available_trained_models(
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
):
    user_models_dir = _safe_child_path(user_id) if user_id else MODEL_ROOT
    available_models = []
    if user_models_dir.exists():
        for model_path in sorted(user_models_dir.rglob("model.pt")):
            model_dir = model_path.parent
            entry = {
                # Overwritten by the summary below with the model's alias where
                # it has one; the folder name is only the fallback for a model
                # a user trained themselves.
                "name": model_dir.name,
                "namespace": str(model_dir.parent.relative_to(MODEL_ROOT))
                if model_dir.parent != MODEL_ROOT
                else None,
            }
            # Summarise what this model can actually do, so a client can build a
            # picker without one /model-info round-trip per model. Absent for
            # models that predate the capability evaluation - clients should
            # treat a missing summary as "only the trained length is known good".
            entry.update(_capability_summary(model_dir))
            available_models.append(entry)
    return available_models


@app.get("/real-data", tags=["Synthetic Data Generation"])
async def get_real_training_data(
    model_name: str = Query("test_model", description="The name of the model"),
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
    max_points: int = Query(
        5000, ge=96, le=50000, description="Cap on returned points; the series is truncated, not resampled."
    ),
) -> dict:
    """The real series a model was trained on, for comparison against its output.

    Served from train_data.csv, which the pretrainer writes next to the model.
    Values are average power in W on the model's own grid - already converted
    from the raw cumulative counter, so they are directly comparable with
    generated output.
    """
    folder_path = _model_path(model_name, user_id)
    train_csv = folder_path / "train_data.csv"
    if not train_csv.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training data stored for model '{model_name}'.",
        )
    metadata = anonymize.load_metadata(folder_path)
    frame = pd.read_csv(train_csv)
    value_columns = [c for c in frame.columns if c != "datetime"]
    if not value_columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="train_data.csv has no value column.",
        )
    column = value_columns[0]
    resolution = metadata.get("resolution_minutes")

    frame = frame.head(max_points)
    has_datetime = "datetime" in frame and len(frame) > 0
    # The shape of the real series is what makes the comparison useful and is
    # safe to publish; the date it was recorded on is not, so a pre-trained
    # model's series is re-dated to the same "now" the generated one gets.
    anonymous_start = anonymize.generation_start(metadata)
    if anonymous_start is not None:
        start = str(anonymous_start)
    else:
        start = str(frame["datetime"].iloc[0]) if has_datetime else None

    return {
        "model": anonymize.public_name(MODEL_ROOT, folder_path, metadata),
        "column": column,
        "resolution_minutes": resolution,
        "n_points": int(len(frame)),
        "start": start,
        "values": [float(v) for v in frame[column].to_numpy()],
    }


@app.get("/model-info", tags=["Synthetic Data Generation"])
async def get_model_info(
    model_name: str = Query("test_model", description="The name of the model"),
    user_id: Optional[str] = Query(
        None,
        description="Optional namespace for legacy models stored as models/{user_id}/{model_name}",
    ),
) -> dict:
    folder_path = _model_path(model_name, user_id)
    if not (folder_path / "model.pt").exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pre-trained model '{model_name}' not found",
        )

    metadata = anonymize.load_metadata(folder_path)
    info = {
        "name": anonymize.public_name(MODEL_ROOT, folder_path, metadata),
        "namespace": str(folder_path.parent.relative_to(MODEL_ROOT))
        if folder_path.parent != MODEL_ROOT
        else None,
    }
    if metadata:
        # metadata.json holds the raw-fetch block, the source path and the
        # training timestamps verbatim; none of that may be served for a model
        # trained on a real meter.
        info["metadata"] = anonymize.public_metadata(MODEL_ROOT, folder_path, metadata)
    return info


if __name__ == "__main__":
    port = int(os.getenv("ML_GRETEL_PORT", 600))
    reload = os.getenv("ML_GRETEL_RELOAD", "True").lower() == "true"

    print(f"Starting server. Go to http://127.0.0.1:{port}/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
