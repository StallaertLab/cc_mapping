from setuptools import setup

setup(
    name='cc_mapping',
    version='0.0.2',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
    ],
)