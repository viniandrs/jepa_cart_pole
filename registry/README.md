# Model Registry

Executed notebooks of superseded model versions, kept so that results can be compared across versions and cited in the thesis.

## Convention

- Whenever a model is improved, the **executed** notebooks of the current version are copied here **before** they are re-run for the new version. `notebooks/` always holds the current model.
- One folder per superseded version: `vN-<short-name>/` (e.g. `v1-vae-z32`), mirroring the `notebooks/` layout. Only the notebooks of stages that depend on the model are copied (data collection is model-independent).
- Each folder has a `README.md` with the date, the configuration, the key results and the reason the version was superseded.
- Checkpoints and data are **not** archived (they are large and gitignored). The notebooks are records of the outputs: they are not guaranteed to re-run against later versions of the `jepacartpole` package.

## Versions

| Version | Encoder | Superseded because | Date |
|---|---|---|---|
