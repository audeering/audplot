Contributing
============

Everyone is invited to contribute to this project.
Feel free to create a `pull request`_ .
If you find errors,
omissions,
inconsistencies,
or other things
that need improvement,
please create an issue_.

.. _issue: https://github.com/audeering/audplot/issues/new/
.. _pull request: https://github.com/audeering/audplot/compare/


Development Installation
------------------------

Instead of pip-installing the latest release from PyPI_,
you should get the newest development version from Github_::

   git clone https://github.com/audeering/audplot/
   cd audplot
   uv sync


This will create a virtual environment
and install the package and all its development dependencies.

To run commands in the virtual environment created by uv,
you can use ``uv run`` or activate the environment directly::

   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows

This way,
your installation always stays up-to-date,
even if you pull new changes from the Github repository.

.. _PyPI: https://pypi.org/project/audplot/
.. _Github: https://github.com/audeering/audplot/


Coding Convention
-----------------

We follow the PEP8_ convention for Python code
and use ruff_ as a linter and code formatter.
In addition,
we check for common spelling errors with codespell_.
Both tools and possible exceptions
are defined in :file:`pyproject.toml`.

The checks are executed in the CI using `pre-commit`_.
You can enable those checks locally by executing::

    uvx pre-commit install
    uvx pre-commit run --all-files

Afterwards ruff_ and codespell_ are executed
every time you create a commit.

You can also call ruff_ and codespell_ directly with uv::

    uvx ruff check --fix .  # lint all Python files, and fix any fixable errors
    uvx ruff format .  # format code of all Python files
    uvx codespell

It can be restricted to specific folders::

    uvx ruff check audplot/ tests/
    uvx codespell audplot/ tests/


.. _codespell: https://github.com/codespell-project/codespell/
.. _PEP8: http://www.python.org/dev/peps/pep-0008/
.. _pre-commit: https://pre-commit.com
.. _ruff: https://beta.ruff.rs


Building the Documentation
--------------------------

If you make changes to the documentation,
you can re-create the HTML pages using Sphinx_.
The necessary packages are already included
in the dev dependency group.

To create the HTML pages, use::

   uv run python -m sphinx docs/ build/sphinx/html -b html

The generated files will be available
in the directory :file:`build/sphinx/html/`.

It is also possible to automatically check if all links are still valid::

   uv run python -m sphinx docs/ build/sphinx/html -b linkcheck

.. _Sphinx: http://sphinx-doc.org


Running the Tests
-----------------

You'll need pytest_ for that,
which is already included in the dev dependency group.

To execute the tests, simply run::

   uv run pytest

.. _pytest: https://pytest.org


Creating a New Release
----------------------

New releases are made using the following steps:

#. Update ``CHANGELOG.rst``
#. Commit those changes as "Release X.Y.Z"
#. Create an (annotated) tag with ``git tag -a X.Y.Z``
#. Push the commit and the tag to Github
