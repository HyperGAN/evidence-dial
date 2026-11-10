"""Stage only the four declared, pinned source repositories."""
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
lock=json.loads((root/'configs/training-sources.lock.json').read_text())['kits']
urls={'CuratorKIT':'https://github.com/Lexsi-Labs/CuratorKIT.git','AuditKIT':'https://github.com/Lexsi-Labs/AuditKIT.git','aligntune':'https://github.com/Lexsi-Labs/aligntune.git','particle-sliders':'https://github.com/HyperGAN/particle-sliders.git'}
for name,url in urls.items():
    dest=root/'vendor'/name
    if dest.exists():
        actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        if actual!=lock[name]:raise ValueError('Existing source differs; preserve it and use a new directory')
    else:
        subprocess.run(['git','clone','--filter=blob:none',url,str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',lock[name]],check=True)
print('All pinned source heads verified')
