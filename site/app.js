import {makePoints, mergeTile, fullSearch} from './search.mjs';
const $ = id => document.getElementById(id);
const points = makePoints();
const canvas = $('plot'), ctx = canvas.getContext('2d');
let state = {scanned:0,best:[],activeStart:-1,running:false}, timer=null;
const query = () => ({x:Number($('query-x').value),y:Number($('query-y').value)});
const tileSize = () => Number($('tile').value);
const kValue = () => Number($('k').value);
function stop() {clearTimeout(timer);timer=null;state.running=false;$('run').textContent=state.scanned===points.length?'Run again →':state.scanned?'Continue scan →':'Run full scan →';}
function reset() {stop();state={scanned:0,best:[],activeStart:-1,running:false};$('verification').textContent='Awaiting scan';$('run').textContent='Run full scan →';$('step').disabled=false;update();}
function draw() {
  const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height,pad=28,x=v=>pad+v*(w-pad*2),y=v=>h-pad-v*(h-pad*2);
  ctx.clearRect(0,0,w,h);ctx.strokeStyle='#e7eadd';ctx.lineWidth=1;
  for(let t=0;t<=10;t++){const v=t/10;ctx.beginPath();ctx.moveTo(x(v),y(0));ctx.lineTo(x(v),y(1));ctx.stroke();ctx.beginPath();ctx.moveTo(x(0),y(v));ctx.lineTo(x(1),y(v));ctx.stroke();}
  const bestIds=new Set(state.best.map(p=>p.id));
  for(const p of points){const active=p.id>=state.activeStart&&p.id<state.scanned&&state.activeStart>=0;
    ctx.fillStyle=bestIds.has(p.id)?'#7e9d27':active?'#4f9cb5':p.id<state.scanned?'#829187':'#c4cdbf';
    ctx.beginPath();ctx.arc(x(p.x),y(p.y),bestIds.has(p.id)?4.8:active?3.2:2.6,0,Math.PI*2);ctx.fill();
  }
  const q=query();
  for(const p of state.best){const v=points[p.id];ctx.strokeStyle='#91a55f70';ctx.lineWidth=1;ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(x(q.x),y(q.y));ctx.lineTo(x(v.x),y(v.y));ctx.stroke();}ctx.setLineDash([]);
  ctx.strokeStyle='#b95532';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x(q.x),y(q.y),9,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#b95532';ctx.beginPath();ctx.arc(x(q.x),y(q.y),4,0,Math.PI*2);ctx.fill();
  ctx.font='10px monospace';ctx.fillStyle='#9b4b30';ctx.fillText('QUERY',x(q.x)+14> w-55?x(q.x)-52:x(q.x)+14, y(q.y)+4);
}
function update(){
  $('qx-value').textContent=query().x.toFixed(2);$('qy-value').textContent=query().y.toFixed(2);
  $('scanned').replaceChildren(document.createTextNode(`${state.scanned} `));const small=document.createElement('small');small.textContent='/ 512';$('scanned').append(small);
  $('progress').value=state.scanned;
  const bytes=tileSize()*2*8;$('memory').textContent=bytes>=1024?`${bytes/1024} KiB`:`${bytes} B`;
  $('memory-caption').textContent=`Two 1 × ${tileSize()} float64 tile buffers`;
  $('scan-state').textContent=state.scanned===points.length?'Complete':state.running?'Scanning':state.scanned?'Paused':'Ready';
  $('tiles').replaceChildren(...Array.from({length:Math.ceil(points.length/tileSize())},(_,i)=>{const e=document.createElement('span');e.className=(i*tileSize()<state.scanned?'done':'')+(i*tileSize()===state.activeStart?' current':'');e.title=`Tile ${i+1}: IDs ${i*tileSize()}–${Math.min(points.length,(i+1)*tileSize())-1}`;return e;}));
  if(state.best.length){$('neighbors').replaceChildren(...state.best.map(p=>{const e=document.createElement('span');e.className='neighbor';e.textContent=`#${String(p.id).padStart(3,'0')} · ${p.distance.toFixed(5)}`;return e;}));}else{$('neighbors').textContent='Run the scan to inspect retained vector IDs and squared distances.';}
  draw();
}
function step(){
  if(state.scanned>=points.length)return;
  state.activeStart=state.scanned;const end=Math.min(points.length,state.scanned+tileSize());
  state.best=mergeTile(state.best,points.slice(state.scanned,end),query(),kValue());state.scanned=end;
  if(end===points.length){const expected=fullSearch(points,query(),kValue());const valid=JSON.stringify(expected)===JSON.stringify(state.best);$('verification').textContent=valid?'Match verified ✓':'Mismatch detected';stop();$('step').disabled=true;}
  update();
}
function animate(){step();if(state.running)timer=setTimeout(animate,window.matchMedia('(prefers-reduced-motion: reduce)').matches?0:150);}
$('run').addEventListener('click',()=>{if(state.running){stop();update();return;}if(state.scanned===points.length)reset();state.running=true;$('run').textContent='Pause scan Ⅱ';animate();});
$('step').addEventListener('click',()=>{stop();step();if(state.scanned<points.length)$('run').textContent='Continue scan →';});
$('reset').addEventListener('click',reset);
for(const id of ['tile','k','query-x','query-y'])$(id).addEventListener('input',reset);
canvas.addEventListener('click',e=>{const r=canvas.getBoundingClientRect();$('query-x').value=Math.max(0,Math.min(1,(e.clientX-r.left-28)/(r.width-56))).toFixed(2);$('query-y').value=Math.max(0,Math.min(1,1-(e.clientY-r.top-28)/(r.height-56))).toFixed(2);reset();});
new ResizeObserver(draw).observe(canvas);
reset();
async function loadBenchmarks(){
  try{const response=await fetch('benchmark.json');if(!response.ok)throw new Error('No published report');const data=await response.json();
    const cases=[...new Set(data.rows.map(r=>r.case))];$('case-select').replaceChildren(...cases.map(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;return o;}));$('case-select').value=cases.at(-1);
    const render=()=>{const rows=data.rows.filter(r=>r.case===$('case-select').value),max=Math.max(...rows.map(r=>r.p50_ms));const head=document.createElement('div');head.className='table-head';const h1=document.createElement('span'),h2=document.createElement('span');h1.textContent='Backend';h2.textContent='Median latency · lower is better';head.append(h1,h2);$('benchmark-table').replaceChildren(head,...rows.map(r=>{const e=document.createElement('div');e.className=`benchmark-row ${r.backend}`;const name=document.createElement('span');name.textContent=r.backend;const track=document.createElement('div');track.className='bar-track';const bar=document.createElement('div');bar.className='bar';bar.style.width=`${r.p50_ms/max*100}%`;track.append(bar);const number=document.createElement('span');number.className='timing';number.textContent=`${r.p50_ms.toFixed(2)} ms`;number.title=`p95: ${r.p95_ms.toFixed(2)} ms; ${r.raw_ms.length} repeats`;e.append(name,track,number);return e;}));
      const n=rows.find(r=>r.backend==='native');$('hardware').textContent=`${data.environment.cpu} · ${data.environment.os} ${data.environment.machine} · float64 · ${data.methodology.repeats} repeats · ${data.methodology.warmups} warmups${n?` · native requests ${n.requested_threads} threads`:''}. p95 and every raw timing are in the JSON. Distance workspace estimates are not peak RSS. Recorded ${data.recorded_at_utc.slice(0,10)}.`;};
    $('case-select').addEventListener('change',render);render();
  }catch{$('benchmark-table').textContent='Recorded measurements could not be loaded. Read the raw report in the repository.';$('case-select').disabled=true;}
}
loadBenchmarks();
