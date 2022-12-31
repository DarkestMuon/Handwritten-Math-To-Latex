from classifier import *

# converts class name given from model to their equivalent latex form
latex_key = {
  'X' : 'x',
  'Delta' : '\delta',
  'alpha' : '\\alpha', 
  'ascii_124' : '|', 
  'beta' : '\\beta', 
  'cos' : '\cos',
  'div' : '\div',
  'exists' : '\exists', 
  'forall' : '\\forall', 
  'forward_slash' : '/', 
  'gamma' : '\gamma', 
  'geq' : '\geq', 
  'gt' : '\gt', 
  'in' : '\in', 
  'infty' : '\infty', 
  'int' : '\int', 
  'lambda' : '\lambda', 
  'ldots' : '\hdots', 
  'leq' : '\leq', 
  'lim' : '\lim', 
  'log' : '\log', 
  'lt' : '\lt', 
  'mu' : '\mu', 
  'neq' : '\\neq', 
  'phi' : '\phi',
  'pi' : '\pi',
  'pm' : '\pi',
  'prime' : '^\prime',
  'rightarrow' : '\\rightarrow',
  'sigma' : '\sigma',
  'sin' : '\sin',
  'sqrt' : '\sqrt',
  'sum' : '\sum',
  'tan' : '\\tan',
  'theta' : '\\theta',
  'times' : '\\times',
  '{' : '\left{',
  '}' : '\rright}',
  '(' : '\left(', 
  ')' : '\\right)',
  '[' : '\left[', 
  ']' : '\\right]'
}

def parser(symbols):
  
  latex_string = ""

  # checks if symbol needs to be converted into its latex form and proceeds accordingly
  for symbol in symbols:

    if symbol in latex_key:
      latex_string += latex_key[symbol] + ' '
      
    else:
      latex_string += symbol + ' '

  return latex_string
