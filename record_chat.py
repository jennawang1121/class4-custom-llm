"""Record actual terminal I/O from the supplied chat.py as asciicast v2."""
import json, time, sys
from pathlib import Path
import pexpect
root=Path(__file__).resolve().parent
out=root/'evidence';out.mkdir(exist_ok=True)
model='llm_runs/20260922T001305_992994Z/model.pt'
child=pexpect.spawn(sys.executable,['-u','chat.py','--model',model,'--transcript','evidence/chat_transcript.json'],cwd=str(root),encoding='utf-8',timeout=60,dimensions=(30,110))
start=time.monotonic()
with (out/'chat_session.cast').open('w') as cast,(out/'chat_terminal.txt').open('w') as plain:
 cast.write(json.dumps({'version':2,'width':110,'height':30,'timestamp':int(time.time()),'title':'Actual expanded nanoGPT chat: 3000 steps, learning rate 0.0015','env':{'TERM':'xterm-256color'}})+'\n')
 class Recorder:
  def write(self,data):
   cast.write(json.dumps([round(time.monotonic()-start,6),'o',data])+'\n');cast.flush();plain.write(data);plain.flush()
  def flush(self):pass
 child.logfile_read=Recorder()
 for prompt in ['the customer','last monday she','Can you explain quantum computing?']:
  child.expect_exact('You: ')
  child.sendline(prompt)
 child.expect_exact('You: ')
 child.sendline('/quit')
 child.expect(pexpect.EOF)
 child.close()
 if child.exitstatus != 0:raise RuntimeError(f'Chat failed: {child.exitstatus}')
print((out/'chat_terminal.txt').read_text())
