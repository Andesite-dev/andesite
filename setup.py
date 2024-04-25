import os
import sys
from os.path import dirname, join as pjoin
from setuptools import setup, find_packages, Command
from setuptools.command.test import test as TestCommand
import version

meta = {}
with open(pjoin('andesite', '__version__.py')) as f:
    exec(f.read(), meta)

# class Publish(Command):
#     """Publish to PyPI with twine."""
#     user_options = []

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def run(self):
#         os.system('python setup.py sdist bdist_wheel')

#         sdist = 'dist/icecream-%s.tar.gz' % meta['__version__']
#         wheel = 'dist/icecream-%s-py2.py3-none-any.whl' % meta['__version__']
#         rc = os.system('twine upload "%s" "%s"' % (sdist, wheel))

#         sys.exit(rc)


class RunTests(TestCommand):
    """
    Run the unit tests.

    By default, `python setup.py test` fails if tests/ isn't a Python
    module (that is, if the tests/ directory doesn't contain an
    __init__.py file). But the tests/ directory shouldn't contain an
    __init__.py file and tests/ shouldn't be a Python module. See

      http://doc.pytest.org/en/latest/goodpractices.html

    Running the unit tests manually here enables `python setup.py test`
    without tests/ being a Python module.
    """
    def run_tests(self):
        from unittest import TestLoader, TextTestRunner
        tests_dir = pjoin(dirname(__file__), 'tests')
        suite = TestLoader().discover(tests_dir)
        result = TextTestRunner().run(suite)
        sys.exit(0 if result.wasSuccessful() else -1)


with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name = 'andesite',
    version = version.get_git_version(),
    author = 'ANDESITE SpA',
    author_email = 'dev@andesite.cl',
    description = 'andesite by ANDESITE SpA',
    keywords = "analytics data technology mining geostatistic estimation simulation software",
    url = "http://www.andesite.cl/",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    include_package_data=True,
    package_dir={'andesite': 'andesite'},
    packages=find_packages(include=['andesite*']),
    # package_dir={"": 'src/andesite'},
    # packages=find_packages(where="src/andesite", exclude=[".DS_Store", "__pycache__"]),
    package_data={'andesite': ["utils/RELEASE-VERSION",
                               "utils/bin/*"]},
    python_requires='>=3.9',
    # entry_points={
    #     'console_scripts': ['andesite = andesite.main:program.run']
    # },
    install_requires = [
        "Cython"
        ],
    # cmdclass={
    #     'test': RunTests,
    #     'publish': Publish,
    # },
)
