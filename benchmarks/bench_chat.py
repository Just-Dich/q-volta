import urllib.request,json,time,pathlib,sys
label=sys.argv[1]; rows=[]
prompts=[('code','Напиши на Python функцию binary_search для отсортированного списка. Добавь короткий пример вызова. Только код.'),('prose','Объясни простыми словами, почему сменяются времена года на Земле. Расскажи о наклоне оси и разнице между полушариями.')]
for kind,prompt in prompts:
 payload={'model':'qwen3.8-27b','messages':[{'role':'user','content':prompt}],'max_tokens':128,'temperature':0,'stream':True,'chat_template_kwargs':{'enable_thinking':False}}
 start=time.monotonic();first=None;parts=[];timings=None;err=None
 try:
  with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=240) as r:
   for line in r:
    if not line.startswith(b'data: '):continue
    raw=line[6:].strip()
    if raw==b'[DONE]':break
    d=json.loads(raw)
    if 'error' in d:err=d['error']
    if 'timings' in d:timings=d['timings']
    for c in d.get('choices',[]):
     text=c.get('delta',{}).get('content','')
     if text and first is None:first=time.monotonic()-start
     parts.append(text)
 except Exception as e:err=repr(e)
 row={'kind':kind,'ttft':first,'elapsed':time.monotonic()-start,'timings':timings,'text':''.join(parts),'error':err};rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
pathlib.Path('/workspace/q-volta/benchmarks/baselines/ninfer-'+label+'.json').write_text(json.dumps({'label':label,'rows':rows},ensure_ascii=False,indent=2))
