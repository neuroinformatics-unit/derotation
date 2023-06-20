from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("derotation")
except PackageNotFoundError:
    # package is not installed
    pass
