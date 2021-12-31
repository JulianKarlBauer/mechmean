# Instruction

## Test installed packages

Run pytest directly

```
pytest --verbose
```

## Test source code in files

Add current dir to path and run pytest

```
python -m pytest
```

## Debug
Start interactive python session for failing tests with

```zsh
pytest --verbose --pdb
```

## Exclude marked tests

```zsh
pytest -m "not scripts"
```

## Note on Globals

<https://stackoverflow.com/a/13300775/8935243>

Current global namespace can be made available in IPython script-execution with

```ipython
%run -i script
```

and in runpy script-execution with

```
runpy.run_path('test.py', init_globals={'a': 16})
```
