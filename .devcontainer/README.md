# Running Iridium-1 in GitHub Codespaces

Open the repository in a Codespace and the container installs CPU torch, then
starts the probe server on port 8080 and forwards it. The 34 M trained rung
loads from `serve/weights/` in under a second; the 1.00 B rung builds from its
config on first request in about 20 s and needs roughly 6 GB of RAM, so ask for
a 4-core / 16 GB machine if you want it (`hostRequirements` already does).

    pytest -q                                   # 211 tests
    python -m iridium ladder                    # the scaling ladder
    PYTHONPATH=. python serve/server.py         # the probe, port 8080
    PYTHONPATH=. python experiments/run_test1b.py   # re-measure the 1 B rung

Codespaces is the better fit than a serverless host for this: the model needs a
process that stays warm and several GB of resident memory, which is exactly what
a serverless function does not give you.
