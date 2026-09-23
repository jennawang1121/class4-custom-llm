"""Original synthetic teaching material; never reads the evaluation suite."""
from pathlib import Path
root = Path(__file__).resolve().parent / 'corpus'
root.mkdir(exist_ok=True)
grammar = []
subjects = [('a bird','birds'),('a dog','dogs'),('a teacher','teachers'),('a driver','drivers'),('a farmer','farmers'),('a child','children'),('one student','students'),('one nurse','nurses')]
states = ['ready','quiet','happy','outside','inside','near the school','beside the garden','at the station']
for singular,plural in subjects:
 for state in states:
  grammar += [f'{singular} is {state} today .', f'{plural} are {state} today .',f'{singular} was {state} yesterday .',f'{plural} were {state} yesterday .']
for state in states:
 grammar += [f'i am {state} today .',f'we are {state} today .',f'she is {state} today .',f'he was {state} yesterday .']
for place in ['school','market','station','garden','hospital','store','office','bank']:
 for singular in ['he','she','the teacher','the driver','the farmer','the nurse']:
  grammar += [f'every morning {singular} walks to the {place} .',f'last monday {singular} walked to the {place} .',f'{singular} is walking to the {place} now .']
 for plural in ['we','they','students','dogs']:
  grammar += [f'every morning {plural} walk to the {place} .',f'{plural} walked to the {place} yesterday .',f'{plural} are walking to the {place} now .']
for name in ['ruth','ben','iris','sam']:
 for action in ['played','worked','rested','talked','jumped','walked']:
  grammar += [f'yesterday {name} {action} near the garden .',f'{name} {action} at the school last monday .']
spatial=[]
# Single-sentence descriptions and varied relational wording, not test stories.
objects=['cup','pencil','toy','shoe','coin','flower','plate','ball','book','lamp']
containers=['basket','drawer','bag','box','cabinet']
for obj in objects:
 for container in containers:
  spatial += [f'a {container} contains a small {obj} .',f'look inside a {container} to find a {obj} .',f'we placed a {obj} inside a {container} .',f'outside a {container} sits a {obj} .']
for high in ['picture','clock','shelf','window','lamp']:
 for low in ['chair','table','bed','desk','basket']:
  spatial += [f'a {high} hangs above a {low} .',f'below a {high} stands a {low} .',f'looking up from a {low} we see a {high} above it .',f'looking down from a {high} we see a {low} below it .']
for a in ['chair','plant','basket','table','cabinet']:
 for b in ['door','window','desk','shelf','box']:
  spatial += [f'a {a} stands to the left of a {b} .',f'from a {a} look right to see a {b} .',f'a {a} stands to the right of a {b} .',f'from a {a} look left to see a {b} .',f'a {a} is beside a {b} .']
for place in ['school','garden','market','hospital','station']:
 spatial += [f'we walked north from the {place} .',f'we walked south toward the {place} .',f'one map shows the {place} to the north .',f'another map shows the {place} to the south .']
for filename,lines in [('grammar.txt',grammar),('spatial_relations.txt',spatial)]:
 assert len(lines)==len(set(lines))
 (root/filename).write_text('\n'.join(lines)+'\n')
 print(filename,len(lines),'unique teaching passages')
