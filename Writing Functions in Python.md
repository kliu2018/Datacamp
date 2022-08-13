# Best Practices
## Crafting a docstring
~~~
# Add a docstring to count_letter()
def count_letter(content, letter):
  """Count the number of times `letter` appears in `content`"""
  if (not isinstance(letter, str)) or len(letter) != 1:
    raise ValueError('`letter` must be a single character string.')
  return len([char for char in content if char == letter])
~~~
~~~
def count_letter(content, letter):
  """Count the number of times `letter` appears in `content`.

  # Add a Google style arguments section
  Args:
    content (str): The string to search.
    letter (str): The letter to search for.
  """
  if (not isinstance(letter, str)) or len(letter) != 1:
    raise ValueError('`letter` must be a single character string.')
  return len([char for char in content if char == letter])
~~~
~~~
  def count_letter(content, letter):
  """Count the number of times `letter` appears in `content`.

  Args:
    content (str): The string to search.
    letter (str): The letter to search for.

  # Add a returns section
  Returns:
    int
  """
  if (not isinstance(letter, str)) or len(letter) != 1:
    raise ValueError('"letter" must be a single character string.')
  return len([char for char in content if char == letter])
  ~~~
  ~~~
  def count_letter(content, letter):
  """Count the number of times `letter` appears in `content`.

  Args:
    content (str): The string to search.
    letter (str): The letter to search for.

  Returns:
    int

  # Add a section detailing what errors might be raised
  Raises:
    letter: If `letter` is not a one-character string.
  """
  if (not isinstance(letter, str)) or len(letter) != 1:
    raise ValueError('`letter` must be a single character string.')
  return len([char for char in content if char == letter])
  ~~~
  
  ## Retrieving docstrings
  ~~~
  # Get the "count_letter" docstring by using an attribute of the function
docstring = count_letter.__doc__

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))
~~~
~~~
import inspect

# Inspect the count_letter() function to get its docstring
docstring = inspect.getdoc(count_letter)

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))
~~~
~~~
import inspect

def build_tooltip(function):
  """Create a tooltip for any function that shows the
  function's docstring.

  Args:
    function (callable): The function we want a tooltip for.

  Returns:
    str
  """
  # Get the docstring for the "function" argument by using inspect
  docstring = inspect.getdoc(function)
  border = '#' * 28
  return '{}\n{}\n{}'.format(border, docstring, border)

print(build_tooltip(count_letter))
print(build_tooltip(range))
print(build_tooltip(print))
~~~
