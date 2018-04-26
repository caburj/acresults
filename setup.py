from setuptools import setup
import os


def read(fname):
    """
    Utility function to read the README file.
    Used for the long_description.  It's nice, because now 1) we have a top level
    README file and 2) it's easier to type in the README file than to put a raw
    string in below ...
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="acresults",
    version="0.1",
    description="post-processing AquaCrop simulation results",
    author="Joseph Caburnay",
    author_email='joseph.caburnay@kuleuven.be',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="MIT",
    url="",
    long_description=read("README.md"),
    packages=['acresults']
)
