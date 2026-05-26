def cancer(x):
  '''Function to change class labels to binary value of 1 and 0. '''
  ctype = ['SCC','BCC','MEL']
  if x in ctype:
    return('1')
  else:
    return('0')