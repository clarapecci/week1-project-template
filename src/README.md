# Week-1 Group Projects: `src` directory

The `src` directory of the group projects template is where you should put your
Python library if you write one for the project. You don't need to write a
library for your project, but if you generate a lot of code in your notebook,
it's a great idea to put as much of it as you can into a Python library then to
import and use that library in the notebook.

Doing this for the project is straightforward; the steps involved are explained
here. For simplicity, we assume that your library is named `projectlib` (but
you should give your library a more descriptive / better name).

1. Make a directory named after your project in the `src` directory. This
   directory must be the name of the library, e.g., `cd src; mkdir projectlib`.
2. Create an `__init__.py` file in the library's directory; your code should go
   here. You can also create other python files and subdirectories that are
   loaded by your `__init__.py` file. For more information on how libraries are
   typically organized, see [this page](
   https://docs.python.org/3/tutorial/modules.html).
3. Go through the `pyproject.toml` file in the root of this directory and edit
   any relevant entries; the comments in the file will direct you to the
   elements that you will likely need to change.
4. To make sure that your library is available to be imported in Python, make
   sure you're in the repository root directory (where the `pyproject.toml`
   file lives) and run the following command:
   `pip install -e .`
