"""Local-only archive readers. No network calls; source ZIPs are never rewritten."""
from __future__ import annotations
import hashlib,json,os,re,shutil,stat,zipfile
from collections import defaultdict
from pathlib import Path,PurePosixPath

IMG={'.jpg','.jpeg','.png','.bmp','.webp'}
KEEP=IMG|{'.txt','.csv','.json','.jsonl','.md','.html','.htm','.zip'}

class Blocked(RuntimeError): pass

def sha(path,algo='sha256'):
    h=hashlib.new(algo)
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)

def jsonl(p,rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def read_jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        for n,s in enumerate(f,1):
            if s.strip():
                r=json.loads(s)
                if not isinstance(r,dict): raise Blocked('Invalid JSONL object %s:%d'%(p,n))
                yield r

def safe_members(z):
    seen=set();result=[]
    for x in z.infolist():
        p=PurePosixPath(x.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in x.filename or any(':' in q for q in p.parts) or stat.S_ISLNK(x.external_attr>>16) or x.flag_bits&1:
            raise Blocked('Unsafe/encrypted archive member: '+x.filename)
        if str(p) in seen: raise Blocked('Duplicate ZIP member: '+x.filename)
        seen.add(str(p));result.append(x)
    if not result:raise Blocked('Empty ZIP')
    return result

def selected(x):
    p=PurePosixPath(x.filename)
    return not x.is_dir() and (p.suffix.lower() in KEEP or p.name.lower().startswith(('readme','license','copying')))

def extract(archive,dest,inventory_dir,expected_md5=None):
    archive=Path(archive).resolve();dest=Path(dest)
    if not archive.is_file():raise Blocked('Missing local archive: '+str(archive))
    h=sha(archive)
    if expected_md5 and sha(archive,'md5').lower()!=expected_md5.lower():raise Blocked('Official MIR image MD5 mismatch: '+str(archive))
    with zipfile.ZipFile(archive) as z:
        allm=safe_members(z);members=[x for x in allm if selected(x)]
        inv={'archive':str(archive),'sha256':h,'members':len(allm),'selected_members':len(members),
             'uncompressed_selected_bytes':sum(x.file_size for x in members),'image_count':sum(PurePosixPath(x.filename).suffix.lower() in IMG for x in members),
             'top_entries':[x.filename for x in allm[:150]],'metadata_entries':[x.filename for x in members if PurePosixPath(x.filename).suffix.lower() not in IMG][:300]}
        dump(Path(inventory_dir)/(dest.name+'.json'),inv)
        marker=dest/'_EXTRACTED.json'
        if dest.exists():
            if marker.is_file() and json.loads(marker.read_text())['archive_sha256']==h:
                # Do not silently trust a stale completion marker after local deletions.
                if all(dest.joinpath(*PurePosixPath(x.filename).parts).is_file() and dest.joinpath(*PurePosixPath(x.filename).parts).stat().st_size==x.file_size for x in members):return dest
            raise Blocked('Existing extraction changed/incomplete; preserve it, use a new extraction root: '+str(dest))
        required=inv['uncompressed_selected_bytes']+512*1024**2
        dest.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(dest.parent).free<required:raise Blocked('Insufficient free space for '+str(archive))
        work=dest.with_name(dest.name+'.extracting')
        if work.exists():raise Blocked('Incomplete extraction preserved: '+str(work))
        work.mkdir()
        for n,x in enumerate(members,1):
            out=work.joinpath(*PurePosixPath(x.filename).parts);out.parent.mkdir(parents=True,exist_ok=True)
            with z.open(x) as src,out.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            if n%5000==0:print('EXTRACT',archive.name,n,'/',len(members),flush=True)
        dump(work/'_EXTRACTED.json',{'archive_sha256':h,'selected_members':len(members),'crc':'checked for extracted members during reading'})
        work.rename(dest)
    return dest

def expand_nested(root,inventory,depth=0):
    if depth>=2:return
    for p in list(Path(root).rglob('*.zip')):
        if '_nested' in p.relative_to(root).parts:continue
        if p.stat().st_size>80*1024**3:raise Blocked('Unexpectedly large nested ZIP: '+str(p))
        dest=Path(root)/'_nested'/(p.stem+'_'+hashlib.sha256(str(p.relative_to(root)).encode()).hexdigest()[:10])
        extract(p,dest,inventory);expand_nested(dest,inventory,depth+1)

def unique_file(root,name,optional=False):
    paths=[p for p in Path(root).rglob('*') if p.is_file() and p.name.lower()==name.lower()]
    if not paths and optional:return None
    if not paths:raise Blocked('Missing '+name+' under '+str(root))
    hs={sha(p) for p in paths}
    if len(hs)>1:raise Blocked('Ambiguous different copies of '+name+': '+str(paths[:10]))
    return sorted(paths,key=lambda p:(len(p.parts),str(p)))[0]

class ImageIndex:
    def __init__(self,root):
        self.root=Path(root).resolve();self.suffix=defaultdict(list);self.count=0
        for p in self.root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG:
                self.count+=1;parts=p.relative_to(self.root).parts
                for k in range(1,min(4,len(parts))+1):self.suffix['/'.join(parts[-k:]).lower()].append(p)
    def resolve(self,name):
        parts=PurePosixPath(str(name).replace('\\','/').strip()).parts
        if '..' in parts or not parts:raise Blocked('Unsafe/empty image-list path: '+str(name))
        for k in range(min(4,len(parts)),0,-1):
            found=self.suffix.get('/'.join(parts[-k:]).lower(),[])
            if len(found)==1:return found[0]
            if len(found)>1:raise Blocked('Ambiguous image mapping (no basename guessing): '+str(name))
        return None

def binary_lines(p):
    out=[]
    for n,s in enumerate(Path(p).read_text(encoding='utf-8-sig').splitlines(),1):
        s=s.strip()
        if s not in ('0','1'):raise Blocked('Expected 0/1 each row; blanks cannot be dropped: %s:%d'%(p,n))
        out.append(int(s))
    return out

def image_lines(p):
    out=[]
    for n,s in enumerate(Path(p).read_text(encoding='utf-8-sig').splitlines(),1):
        s=s.strip()
        if not s or PurePosixPath(s.replace('\\','/')).suffix.lower() not in IMG:raise Blocked('Invalid image list line: %s:%d'%(p,n))
        out.append(s)
    if len(set(out))!=len(out):raise Blocked('Duplicate image-list names: '+str(p))
    return out
