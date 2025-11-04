def test_imports() :
    import importlib
    pkg = importlib.import_module("SatoshiRig")
    assert hasattr(pkg , "__all__")


