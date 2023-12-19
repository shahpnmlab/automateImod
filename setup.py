import os

import setuptools
from setuptools import setup

dependency_links = []
install_requires = []
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "requirements.txt"))) as f:
    for line in f:
        if line.startswith("#"):
            continue
        elif line.startswith("--extra-index-url"):
            dependency_links.append(line.strip())
        elif line.startswith("git"):
            line = line.strip()
            package_name = line.split("/")[-1].split("@")[0].split("@")[0]
            install_requires.append(f'{package_name} @ {line}')
        else:
            install_requires.append(line.strip())
setup(name='automateImod',
      version='0.2',
      url='',
      license='',
      author='Pranav NM Shah',
      packages=setuptools.find_packages(),
      install_requires=install_requires,
      dependency_links=dependency_links,
      include_package_data=True,
      entry_points={
          'console_scripts': ['automateImod=automateImod.run:automateImod'],
      },
      zip_safe=False
      )
