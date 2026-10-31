/* Evidence-only client UI. Loads trained model metrics and precomputed test examples. */
let report, samples=[], selected=0, viewMode='resnet';
const $=(id)=>document.getElementById(id);
const pct=(v,n=1)=>v==null?'—':(100*v).toFixed(n)+'%';
const esc=(s)=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function start(){
 try{
  const [a,b]=await Promise.all([fetch('assets/results.json'),fetch('assets/showcase.json')]);
  if(!a.ok||!b.ok)throw Error('Experiment artifacts unavailable');
  report=await a.json();samples=await b.json();
  renderMetrics();renderChart();renderMatrix();renderPicker();renderInspector();
  if(samples.length){$('hero-image').src=samples[0].resnet_image;$('hero-case').textContent='#'+String(samples[0].id).padStart(3,'0');$('hero-prediction').textContent=samples[0].resnet.prediction;$('hero-confidence').textContent=pct(samples[0].resnet.confidence)}
 }catch(e){console.warn(e); $('cnn-accuracy').textContent='Unavailable';$('resnet-accuracy').textContent='Unavailable';}
}
function renderMetrics(){
 const a=report.models.custom_cnn.metrics,b=report.models.rotation_resnet18.metrics;
 $('cnn-accuracy').textContent=pct(a.accuracy);$('resnet-accuracy').textContent=pct(b.accuracy);
 $('cnn-f1').textContent='F1 '+a.f1_macro.toFixed(3);$('resnet-f1').textContent='F1 '+b.f1_macro.toFixed(3);
 $('test-count').innerHTML=esc(b.n_test)+'<span class="sample-suffix"> samples</span>';
 $('cnn-track').style.width=pct(a.accuracy);$('resnet-track').style.width=pct(b.accuracy);
}
function renderChart(){
 const series=[{key:'custom_cnn',stroke:'#e9bc86'},{key:'rotation_resnet18',stroke:'#9eeac7'}];
 const w=520,h=260,pad={l:46,r:25,t:19,b:37};const px=(i,m)=>pad.l+i/(Math.max(1,m-1))*(w-pad.l-pad.r);const py=y=>h-pad.b-(y-.0)*(h-pad.t-pad.b);
 let html=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Validation accuracies on each training epoch">`;
 for(const v of [0,.25,.5,.75,1]) html+=`<line x1="${pad.l}" y1="${py(v)}" x2="${w-pad.r}" y2="${py(v)}" stroke="#25444b" stroke-dasharray="3 4"/><text x="${pad.l-8}" y="${py(v)+4}" text-anchor="end" font-size="10" fill="#85a7aa" font-family="monospace">${Math.round(v*100)}</text>`;
 for(const s of series){let values=report.models[s.key].history.map(x=>x.val_accuracy);let d=values.map((v,i)=>`${i?'L':'M'} ${px(i,values.length).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');html+=`<path d="${d}" fill="none" stroke="${s.stroke}" stroke-width="2.6" stroke-linejoin="round"/>`; values.forEach((v,i)=>html+=`<circle cx="${px(i,values.length)}" cy="${py(v)}" r="2.6" fill="${s.stroke}"/>`);}
 html+=`<text x="${w-38}" y="${h-4}" font-size="10" fill="#8ea9ad" text-anchor="middle" font-family="monospace">EPOCH →</text><text x="10" y="8" font-size="9" fill="#8ea9ad" font-family="monospace">ACC %</text>`;
 html+=`<line x1="${pad.l+13}" y1="${h-19}" x2="${pad.l+33}" y2="${h-19}" stroke="#e9bc86" stroke-width="3"/><text x="${pad.l+39}" y="${h-15}" font-size="10" fill="#b5d0cd" font-family="monospace">Custom CNN</text>`;
 html+=`<line x1="${pad.l+163}" y1="${h-19}" x2="${pad.l+183}" y2="${h-19}" stroke="#9eeac7" stroke-width="3"/><text x="${pad.l+190}" y="${h-15}" font-size="10" fill="#b5d0cd" font-family="monospace">ResNet-18</text>`;
 $('learning-chart').innerHTML=html+'</svg>';
}
function renderMatrix(){
 const data=report.models[$('matrix-model').value].metrics.confusion_matrix;
 const w=430,h=290,left=57,top=10,cell=24.2; const largest=Math.max(...data.flat(),1);
 let s=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Confusion matrix: true rows and predicted columns">`;
 for(let i=0;i<10;i++){
  s+=`<text x="${left-17}" y="${top+i*cell+cell*.67}" font-size="10" text-anchor="middle" fill="#9eb8b9" font-family="monospace">${i}</text>`;
  s+=`<text x="${left+i*cell+cell*.5}" y="${top+10*cell+17}" font-size="10" text-anchor="middle" fill="#9eb8b9" font-family="monospace">${i}</text>`;
  for(let j=0;j<10;j++){let v=data[i][j],ratio=v/largest;let fill=`rgba(146,231,192,${(.075+ratio*.88).toFixed(3)})`;
   s+=`<rect x="${left+j*cell}" y="${top+i*cell}" width="${cell-2}" height="${cell-2}" rx="2" fill="${fill}"/><text x="${left+j*cell+(cell-2)/2}" y="${top+i*cell+cell*.64}" font-size="9" text-anchor="middle" fill="${ratio>.45?'#06241e':'#9ebab8'}" font-family="monospace">${v}</text>`;}
 }
 s+=`<text x="${left+cell*5}" y="${h-4}" font-size="9" text-anchor="middle" fill="#7da29a" font-family="monospace">PREDICTED CLASS →</text><text x="12" y="${top+cell*5}" font-size="9" text-anchor="middle" fill="#7da29a" transform="rotate(-90 12 ${top+cell*5})" font-family="monospace">TRUE LABEL</text></svg>`;
 $('confusion-chart').innerHTML=s;
}
function renderPicker(){
 $('sample-picker').innerHTML=samples.map((item,i)=>`<button class="sample-card ${i===selected?'active':''}" data-index="${i}" aria-label="Inspect test sample ${item.id}, digit ${item.truth}" aria-pressed="${i===selected}"><img src="${esc(item.original_image)}" alt="Digit ${item.truth}"><div class="sample-card-meta"><span>#${String(item.id).padStart(3,'0')}</span><span class="${item.resnet.prediction===item.truth?'correct':'incorrect'}">${item.resnet.prediction===item.truth?'✓ MATCH':'× ERROR'}</span></div></button>`).join('');
 document.querySelectorAll('.sample-card').forEach(b=>b.addEventListener('click',()=>{selected=Number(b.dataset.index);renderPicker();renderInspector();}));
}
function renderInspector(){
 if(!samples.length)return;
 let s=samples[selected], model=viewMode==='cnn'?s.cnn:s.resnet;
 let url=viewMode==='original'?s.original_image:viewMode==='cnn'?s.cnn_image:s.resnet_image;
 $('inspect-image').src=url;$('overlay-stamp').textContent=viewMode==='original'?'ORIGINAL INPUT':'GRAD-CAM / '+(viewMode==='cnn'?'CUSTOM CNN':'RESNET-18');
 $('case-id').textContent='#'+String(s.id).padStart(3,'0');$('view-prediction').textContent=model.prediction;
 $('view-class').textContent=model.prediction;$('view-confidence').textContent=pct(model.confidence);$('view-truth').textContent=s.truth;
 $('view-outcome').textContent=model.prediction===s.truth?'✓ Correct':'× Incorrect';$('view-outcome').style.color=model.prediction===s.truth?'#9eeac7':'#ffb49b';
 $('probability-bars').innerHTML=model.probabilities.map((p,i)=>`<div class="prob-line ${i===model.prediction?'chosen':''}"><span>${i}</span><div class="prob-bg"><i style="width:${Math.max(p*100,.35)}%"></i></div><span>${(p*100).toFixed(1)}%</span></div>`).join('');
 document.querySelectorAll('.view-mode').forEach(b=>{b.classList.toggle('active',b.dataset.mode===viewMode);b.setAttribute('aria-pressed',b.dataset.mode===viewMode)});
}
document.querySelectorAll('.view-mode').forEach(b=>b.addEventListener('click',()=>{viewMode=b.dataset.mode;renderInspector()}));
$('matrix-model').addEventListener('change',renderMatrix);
start();
// Resolve the public GitHub source URL only when published on a github.io project site.
// No invented repository name or owner is ever hardcoded.
(function(){
 const link=document.querySelector('.repo-link');
 if(!link)return;
 const host=window.location.hostname;
 const parts=window.location.pathname.split('/').filter(Boolean);
 if(host.endsWith('.github.io') && parts.length){
   const owner=host.split('.')[0],repo=parts[0];
   link.href='https://github.com/'+encodeURIComponent(owner)+'/'+encodeURIComponent(repo)+'/blob/main/README.md';
   link.target='_blank';link.rel='noopener noreferrer';
 }else{
   link.href='#methodology';
   link.innerHTML='Explore project methodology <span>↗</span>';
 }
})();
