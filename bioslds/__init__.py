from pkg_resources import get_distribution

dist_name = __name__
__version__ = get_distribution(dist_name).version
