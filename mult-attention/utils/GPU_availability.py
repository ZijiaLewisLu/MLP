# GPU_availability returns [least_occupied_GPU, ..., most_occupied_GPU].
# Each element of the list is an GPU index (starting from 0).
# It is ensured that the performance of each GPU in the list is at most P2.
# P0 is the maximum performance, indicating that one GPU is completely occupied.

# Credit to Gaiyu

def GPU_availability():
  import itertools
  from subprocess import Popen, PIPE
  import re
  output = Popen(['nvidia-smi'], stdout=PIPE).communicate()[0]
  lines = output.split('\n')
  performance = {}
  index = 0
  for i in range(len(lines)):
    if 'GTX' in lines[i]:
      p = int(re.search(r'P(\d?\d)', lines[i+1]).group(0)[-1])
      if p>1:
        try:
          performance[p].append(index)
        except:
          performance.update({p : [index]})
      index += 1
  return list(itertools.chain(*[performance[key] for key in reversed(sorted(performance.keys()))]))

if __name__ == '__main__':
  print GPU_availability() 
