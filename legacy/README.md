# Legacy Scripts

This folder contains standalone solver scripts that predate the `MixedHierarchyGames` package (`src/`). They implement their own KKT construction, symbolic variable setup, and PATH solver interface directly, without using the package API.

These scripts are preserved for historical reference and are used by:
- `test/test_python_integration.py` - Python integration test
- `test/test_hardware_nplayer_navigation.py` - Hardware navigation test
- `experiments/three_player_chain_validation.jl` - Validation against old solver

For active experiment code that uses the `MixedHierarchyGames` package, see the `experiments/` folder.
